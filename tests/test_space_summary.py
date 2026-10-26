import unittest

from core.data_table import List_Output_Label, Output_Label
from main import SUBSPACE_TO_SPACE, summarize_space_results


class SpaceSummaryTests(unittest.TestCase):
    def test_summary_covers_all_subspaces_and_rounds_scores(self) -> None:
        labels = List_Output_Label(
            labels=[
                Output_Label(
                    activity="First activity",
                    student_labeled_spaces=["understand"],
                    student_labeled_subspaces=["explore"],
                    result=[1],
                    Reason="Evidence supports explore.",
                ),
                Output_Label(
                    activity="Second activity",
                    student_labeled_spaces=["understand"],
                    student_labeled_subspaces=["explore"],
                    result=[0],
                    Reason="Evidence is inconclusive.",
                )
            ]
        )
        inferred = {
            subspace: (1 if subspace == "empathize" else 0)
            for subspace in SUBSPACE_TO_SPACE
            if subspace != "explore"
        }

        summary = summarize_space_results(labels, inferred)

        all_subspaces = set()
        for space in (
            summary.understand,
            summary.synthesize,
            summary.ideate,
            summary.prototype,
            summary.implement,
        ):
            categorized = space.passed + space.not_enough_evidence + space.incorrect
            self.assertEqual(len(categorized), 4)
            self.assertEqual(len(set(categorized)), 4)
            all_subspaces.update(categorized)

        self.assertEqual(all_subspaces, set(SUBSPACE_TO_SPACE))
        self.assertEqual(summary.understand.passed, ["explore", "empathize"])
        self.assertEqual(
            summary.understand.not_enough_evidence, ["observe", "reflect"]
        )
        self.assertEqual(summary.understand.score, 1)
        self.assertEqual(summary.synthesize.score, 0)
        self.assertEqual(len(all_subspaces), 20)

    def test_space_score_passes_when_any_subspace_passes(self) -> None:
        labels = List_Output_Label(
            labels=[
                Output_Label(
                    activity="Activity with supported label",
                    student_labeled_spaces=["understand"],
                    student_labeled_subspaces=["explore", "observe"],
                    result=[1, -1],
                    Reason="Mixed results.",
                )
            ]
        )
        inferred = {
            subspace: 0
            for subspace in SUBSPACE_TO_SPACE
            if subspace not in {"explore", "observe"}
        }

        summary = summarize_space_results(labels, inferred)

        self.assertEqual(summary.understand.score, 1)
        self.assertEqual(summary.understand.passed, ["explore"])
        self.assertEqual(summary.understand.incorrect, ["observe"])

    def test_space_score_is_negative_only_when_all_subspaces_are_incorrect(self) -> None:
        labels = List_Output_Label(
            labels=[
                Output_Label(
                    activity="Activity with four incorrect labels",
                    student_labeled_spaces=["understand"],
                    student_labeled_subspaces=[
                        "explore",
                        "observe",
                        "empathize",
                        "reflect",
                    ],
                    result=[-1, -1, -1, -1],
                    Reason="No labels are supported.",
                )
            ]
        )
        inferred = {
            subspace: 0
            for subspace in SUBSPACE_TO_SPACE
            if subspace not in {"explore", "observe", "empathize", "reflect"}
        }

        summary = summarize_space_results(labels, inferred)

        self.assertEqual(summary.understand.score, -1)


if __name__ == "__main__":
    unittest.main()
