from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Literal, TypedDict

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.data_table import (
    List_Output_Label,
    List_Student_HCD_Label,
)
from core.postprocessing import FinalProcessing
from core.preprocessing import PreProcessor
from core.processing import Processing
from database.db import (
    fetch_unlabeld_activity,
    label_activity,
    get_activity_annotations,
    get_label_stats,
)
from dotenv import load_dotenv

load_dotenv()


class SpaceSummary(BaseModel):
    score: Literal[-1, 0, 1]
    passed: list[str]
    not_enough_evidence: list[str]
    incorrect: list[str]


class SpaceSummaryResponse(BaseModel):
    understand: SpaceSummary
    synthesize: SpaceSummary
    ideate: SpaceSummary
    prototype: SpaceSummary
    implement: SpaceSummary


class FinalLabelResponse(BaseModel):
    activity: str
    student_labeled_spaces: list[str]
    student_labeled_subspaces: list[str]
    result: list[int]


class FinalLabelsResponse(BaseModel):
    labels: list[FinalLabelResponse]


class ClassificationResponse(BaseModel):
    student_labels: List_Student_HCD_Label
    final_labels: FinalLabelsResponse
    space_summary: SpaceSummaryResponse


class RootResponse(BaseModel):
    message: str
    endpoints: dict[str, str]


class UnlabeledActivityResponse(BaseModel):
    rowid: int | None
    Activity: str | None


class LabelActivityRequest(BaseModel):
    rowid: int
    HCD_Space: str
    HCD_Subspace: str
    Reason: str
    Annotator: str


class LabelActivityResponse(BaseModel):
    success: bool
    message: str
    inserted_new: bool = False


class ActivityAnnotation(BaseModel):
    rowid: int
    Activity: str
    HCD_Space: str
    HCD_Subspace: str
    Reason: str
    Annotator: str


class ActivityGroup(BaseModel):
    activity: str
    count: int
    annotators: list[str]
    annotations: list[ActivityAnnotation]


class ActivityAnnotationsResponse(BaseModel):
    total_activities: int
    groups: list[ActivityGroup]


class LabelStatsResponse(BaseModel):
    total: int
    labeled: int
    unlabeled: int


app = FastAPI(title="SIIP HCD Classifier API", version="0.1.0")

preprocessor = PreProcessor()
processor = Processing()
final_processor = FinalProcessing()


SUBSPACE_TO_SPACE = {
    "explore": "understand",
    "observe": "understand",
    "empathize": "understand",
    "reflect": "understand",
    "debrief": "synthesize",
    "organize": "synthesize",
    "interpret": "synthesize",
    "define": "synthesize",
    "brainstorm": "ideate",
    "propose": "ideate",
    "plan": "ideate",
    "narrow concepts": "ideate",
    "create": "prototype",
    "engage": "prototype",
    "evaluate": "prototype",
    "iterate": "prototype",
    "support": "implement",
    "sustain": "implement",
    "evolve": "implement",
    "execute": "implement",
}


class _SpaceSummaryAccumulator(TypedDict):
    passed: list[str]
    not_enough_evidence: list[str]
    incorrect: list[str]


def summarize_space_results(
    labels: List_Output_Label,
    inferred_labels: dict[str, int],
) -> SpaceSummaryResponse:
    scores_by_subspace: dict[str, list[int]] = {
        subspace: [] for subspace in SUBSPACE_TO_SPACE
    }

    for label in labels.labels:
        if len(label.student_labeled_subspaces) != len(label.result):
            raise ValueError(
                "Final evaluation must return one result for each student subspace."
            )

        for subspace, result in zip(
            label.student_labeled_subspaces, label.result
        ):
            space = SUBSPACE_TO_SPACE.get(subspace.strip().lower())
            if space is None:
                raise ValueError(
                    f"Unknown HCD subspace in final evaluation: {subspace}"
                )
            if result not in (-1, 0, 1):
                raise ValueError(f"Unexpected evaluation result: {result}")

            scores_by_subspace[subspace.strip().lower()].append(result)

    if set(inferred_labels) != {
        subspace for subspace, results in scores_by_subspace.items() if not results
    }:
        raise ValueError(
            "LLM inference must provide exactly the subspaces without student evaluations."
        )

    scores_by_space: dict[str, list[int]] = {
        space: []
        for space in ("understand", "synthesize", "ideate", "prototype", "implement")
    }
    summary_by_space: dict[str, _SpaceSummaryAccumulator] = {
        space: {
            "passed": [],
            "not_enough_evidence": [],
            "incorrect": [],
        }
        for space in scores_by_space
    }

    for subspace, space in SUBSPACE_TO_SPACE.items():
        results = scores_by_subspace[subspace]
        if subspace in inferred_labels:
            result = inferred_labels[subspace]
            if result not in (0, 1):
                raise ValueError(f"Unexpected inferred result for {subspace}: {result}")
        else:
            if 1 in results:
                result = 1
            elif 0 in results:
                result = 0
            else:
                result = -1

        scores_by_space[space].append(result)
        category = {
            1: "passed",
            0: "not_enough_evidence",
            -1: "incorrect",
        }[result]
        summary_by_space[space][category].append(subspace)

    return SpaceSummaryResponse(
        **{
            space: SpaceSummary(
                score=(
                    1
                    if 1 in scores
                    else -1
                    if all(score == -1 for score in scores)
                    else 0
                ),
                **summary_by_space[space],
            )
            for space, scores in scores_by_space.items()
        }
    )


def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content_type = (file.content_type or "").lower()
    if not content_type:
        content_type = (
            "application/pdf" if file.filename.lower().endswith(".pdf") else ""
        )

    if content_type not in {
        "application/pdf",
        "application/octet-stream",
    } and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")


async def _persist_upload(file: UploadFile) -> Path:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    suffix = Path(file.filename or "uploaded.pdf").suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = Path(tmp.name)
        tmp.write(contents)

    await file.close()
    return path


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="SIIP HCD Classifier API",
        endpoints={
            "health": "/health",
            "classify": "/classify",
            "fetch-unlabeled": "/fetch-unlabeled",
            "label-activity": "/label-activity",
            "activity-annotations": "/activity-annotations",
            "label-stats": "/label-stats",
            "docs": "/docs",
        },
    )


@app.post("/classify", response_model=ClassificationResponse)
async def classify_pdf(file: UploadFile = File(...)) -> ClassificationResponse:
    _validate_upload(file)
    temp_path: Path | None = None

    try:
        temp_path = await _persist_upload(file)
        student_labels = await asyncio.to_thread(
            preprocessor.invoke, temp_path.as_posix()
        )
        llm_labels = await processor.aclassify_table(student_labels)
        final_labels = await final_processor.afinal_eval(student_labels, llm_labels)
        evaluated_subspaces = {
            subspace.strip().lower()
            for label in final_labels.labels
            for subspace in label.student_labeled_subspaces
        }
        missing_subspaces = [
            subspace
            for subspace in SUBSPACE_TO_SPACE
            if subspace not in evaluated_subspaces
        ]
        inferred_response = (
            await final_processor.infer_missing_subspaces(
                [entry.activity for entry in student_labels.tables],
                missing_subspaces,
            )
            if missing_subspaces
            else None
        )
        inferred_labels = {
            label.subspace: label.result
            for label in (inferred_response.labels if inferred_response else [])
        }
        if len(inferred_labels) != len(missing_subspaces) or set(inferred_labels) != set(
            missing_subspaces
        ):
            raise ValueError(
                "LLM inference must return each unevaluated subspace exactly once."
            )
        space_results = summarize_space_results(final_labels, inferred_labels)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except openai.APIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass

    return ClassificationResponse(
        student_labels=student_labels,
        final_labels=FinalLabelsResponse(
            labels=[
                FinalLabelResponse(
                    activity=label.activity,
                    student_labeled_spaces=label.student_labeled_spaces,
                    student_labeled_subspaces=label.student_labeled_subspaces,
                    result=label.result,
                )
                for label in final_labels.labels
            ]
        ),
        space_summary=space_results,
    )


@app.get("/fetch-unlabeled", response_model=UnlabeledActivityResponse)
async def fetch_unlabeled() -> UnlabeledActivityResponse:
    """Fetch one unlabeled activity from the database."""
    activity = await fetch_unlabeld_activity()

    if activity is None:
        return UnlabeledActivityResponse(rowid=None, Activity=None)

    return UnlabeledActivityResponse(
        rowid=activity.get("rowid"), Activity=activity.get("Activity")
    )


@app.post("/label-activity", response_model=LabelActivityResponse)
async def label_activity_endpoint(
    request: LabelActivityRequest,
) -> LabelActivityResponse:
    """
    Label an activity with HCD classification.

    If the target row was already labeled by another annotator, a new duplicate
    row is automatically inserted so both annotations are preserved.
    The response field `inserted_new` will be True in that case.
    """
    result = await label_activity(
        activity_id=request.rowid,
        HCD_Space=request.HCD_Space,
        HCD_Subspace=request.HCD_Subspace,
        reason=request.Reason,
        annotator=request.Annotator,
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail="Failed to label activity")

    inserted_new = result["inserted_new"]
    if inserted_new:
        msg = (
            f"Activity {request.rowid} was already labeled. "
            f"A new duplicate entry has been created for annotator '{request.Annotator}'."
        )
    else:
        msg = f"Activity {request.rowid} labeled successfully by '{request.Annotator}'."

    return LabelActivityResponse(success=True, message=msg, inserted_new=inserted_new)


@app.get("/activity-annotations", response_model=ActivityAnnotationsResponse)
async def activity_annotations() -> ActivityAnnotationsResponse:
    """
    Return activities that have been labeled by more than one annotator.

    Only activities with 2+ label entries are included — these are exactly
    the cases where concurrent annotators labeled the same activity and a
    duplicate row was auto-inserted.  Use this to compare and reconcile
    differing annotation results for the same activity.
    """
    rows = await get_activity_annotations()

    # Group by Activity text (SQL already filters to count > 1)
    groups_map: dict[str, list[ActivityAnnotation]] = {}
    for row in rows:
        key = row.get("Activity", "")
        ann = ActivityAnnotation(
            rowid=row.get("rowid"),
            Activity=key,
            HCD_Space=row.get("HCD_Space", ""),
            HCD_Subspace=row.get("HCD_Subspace", ""),
            Reason=row.get("Reason", ""),
            Annotator=row.get("Annotator", ""),
        )
        groups_map.setdefault(key, []).append(ann)

    groups = [
        ActivityGroup(
            activity=act,
            count=len(anns),
            annotators=[a.Annotator for a in anns],
            annotations=anns,
        )
        for act, anns in groups_map.items()
    ]

    return ActivityAnnotationsResponse(
        total_activities=len(groups),
        groups=groups,
    )


@app.get("/label-stats", response_model=LabelStatsResponse)
async def label_stats() -> LabelStatsResponse:
    """
    Return statistics about activity labeling progress.
    """
    stats = await get_label_stats()
    return LabelStatsResponse(**stats)
