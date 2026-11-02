# -*- coding: utf-8 -*-
# Global data table structure for HCD classification

from typing import Literal

from pydantic import BaseModel, Field


class Student_HCD_Label(BaseModel):
    """Structured representation of a single activity row from the HCD table.

    Attributes:
        activity: Text describing the activity from the source table.
        HCD_Spaces: Ordered list of HCD spaces linked to the activity.
        HCD_Subspaces: Ordered list of HCD subspaces linked to the activity.
    """

    activity: str = Field(..., description="The Content in the Activity column")
    HCD_Spaces: list[str] = Field(
        ..., description="The Content in the corresponding activity's HCD Space column"
    )
    HCD_Subspaces: list[str] = Field(
        ...,
        description="The Content in the corresponding activity's HCD Subspace column",
    )


class List_Student_HCD_Label(BaseModel):
    """Container for multiple activity rows returned by the parser.

    Attributes:
        tables: Collection of extracted activity entries as `Student_HCD_Label` records.
    """

    tables: list[Student_HCD_Label] = Field(
        ..., description="List of extracted table data"
    )


class LLM_HCD_Label(BaseModel):
    """Model-assigned HCD labels for a single activity entry.

    Attributes:
        activity: Text describing the activity from the source table.
        HCD_Spaces: Ordered list of HCD spaces linked to the activity.
        HCD_Subspaces: Ordered list of HCD subspaces linked to the activity.
    """

    activity: str = Field(..., description="The Content in the Activity column")
    HCD_Spaces: list[str] = Field(
        ..., 
        description="MUST be exactly ONE of: ['understand', 'synthesize', 'ideate', 'prototype', 'implement']"
    )
    HCD_Subspaces: list[str] = Field(
        ...,
        description="MUST be exactly ONE of: ['explore', 'observe', 'empathize', 'reflect', 'debrief', 'organize', 'interpret', 'define', 'brainstorm', 'propose', 'plan', 'narrow concepts', 'create', 'engage', 'evaluate', 'iterate', 'support', 'sustain', 'evolve', 'execute']",
    )


class Output_Label(BaseModel):
    """Final evaluation result for a student activity label.

    Attributes:
        activity: Text describing the original activity entry.
        student_labeled_spaces: Student-provided spaces as a comma-separated string.
        student_labeled_subspaces: Student-provided subspaces as a comma-separated string.
        result: 1 if correct, 0 if not enough evidence, -1 if incorrect.
        Reason: Short justification for the assigned result.
    """

    activity: str = Field(..., description="The Content in the Activity column")

    student_labeled_spaces: list[str] = Field(
        ..., description="The student's labeled HCD spaces (lowercased, normalized)"
    )
    student_labeled_subspaces: list[str] = Field(
        ..., description="The student's labeled HCD subspaces (lowercased, normalized)"
    )
    result: list[int] = Field(
        ...,
        description=(
            "Per-subspace evaluation results aligned to student_labeled_subspaces. "
            "Each item: 1=correct, 0=not enough evidence, -1=incorrect."
        ),
    )
    Reason: str = Field(
        ..., description="The reason for marking the result as 1, 0 or -1"
    )


class List_Output_Label(BaseModel):
    """Container for multiple output label entries.

    Attributes:
        labels: Collection of extracted output label entries as `Output_Label` records.
    """

    labels: list[Output_Label] = Field(
        ..., description="List of extracted output label data"
    )


class InferredSubspaceLabel(BaseModel):
    """LLM evidence assessment for a subspace with no student evaluation."""

    subspace: Literal[
        "explore",
        "observe",
        "empathize",
        "reflect",
        "debrief",
        "organize",
        "interpret",
        "define",
        "brainstorm",
        "propose",
        "plan",
        "narrow concepts",
        "create",
        "engage",
        "evaluate",
        "iterate",
        "support",
        "sustain",
        "evolve",
        "execute",
    ]
    result: Literal[-1, 1]


class List_InferredSubspaceLabel(BaseModel):
    """LLM assessments for requested subspaces not labeled by the student."""

    labels: list[InferredSubspaceLabel]
