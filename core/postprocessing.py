import asyncio
import os
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import init_chat_model

from core.data_table import (
    Output_Label,
    LLM_HCD_Label,
    List_Student_HCD_Label,
    List_Output_Label,
    List_InferredSubspaceLabel,
)
from core.model_config import FINAL_EVAL_MODEL
from core.prompt import FINAL_EVAL_SYS_PROMPT, REPORT_SUBSPACE_EVAL_PROMPT
from core.utils import KNOWN_SPACES, KNOWN_SUBSPACES, normalize_list


class FinalProcessing:
    """Post-processing helpers for final activity evaluation."""

    def __init__(self) -> None:
        """
        Initialize the FinalProcessing helper.

        Sets up the chat model using the final evaluation configuration and binds it to a structured output schema
        (`Output_Label`) for consistent final activity evaluation results.
        """
        self._model = FINAL_EVAL_MODEL
        self.bound_model = self._model.with_structured_output(Output_Label)
        self.inference_model = self._model.with_structured_output(
            List_InferredSubspaceLabel
        )

    async def assess_report_subspaces(
        self, report_text: str, subspaces: list[str]
    ) -> List_InferredSubspaceLabel:
        return await self.inference_model.ainvoke(
            [
                {"role": "system", "content": REPORT_SUBSPACE_EVAL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Assess every requested subspace:\n{subspaces}\n\n"
                        f"Report evidence:\n{report_text}"
                    ),
                },
            ]
        )

    @staticmethod
    def _build_eval_prompt(student_entry, llm_entry) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FINAL_EVAL_SYS_PROMPT},
            {
                "role": "user",
                "content": (
                    "Evaluate the following student and LLM labels for the activity:\n\n"
                    f"Activity: {student_entry.activity}\n"
                    f"Student HCD Spaces: {', '.join(student_entry.HCD_Spaces)}\n"
                    f"Student HCD Subspaces: {', '.join(student_entry.HCD_Subspaces)}\n"
                    f"LLM HCD Spaces: {', '.join(llm_entry.HCD_Spaces)}\n"
                    f"LLM HCD Subspaces: {', '.join(llm_entry.HCD_Subspaces)}\n"
                ),
            },
        ]

    def final_eval(
        self,
        student_hcd_label: List_Student_HCD_Label,
        llm_hcd_label: list[LLM_HCD_Label],
    ) -> List_Output_Label:
        """Evaluate the final output labels against the student labels."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.afinal_eval(student_hcd_label, llm_hcd_label))

        response = [
            self.bound_model.invoke(self._build_eval_prompt(student_entry, llm_entry))
            for student_entry, llm_entry in zip(student_hcd_label.tables, llm_hcd_label)
        ]
        return List_Output_Label(labels=response)

    async def afinal_eval(
        self,
        student_hcd_label: List_Student_HCD_Label,
        llm_hcd_label: list[LLM_HCD_Label],
        max_concurrency: int = 4,
    ) -> List_Output_Label:
        sem = asyncio.Semaphore(max_concurrency)

        async def evaluate(student_entry, llm_entry):
            async with sem:
                return await self.bound_model.ainvoke(
                    self._build_eval_prompt(student_entry, llm_entry)
                )

        tasks = [
            evaluate(student_entry, llm_entry)
            for student_entry, llm_entry in zip(student_hcd_label.tables, llm_hcd_label)
        ]
        results = await asyncio.gather(*tasks)
        return List_Output_Label(labels=results)

    def display_output_labels(
        self, output_labels: Union[List_Output_Label, list[Output_Label]]
    ) -> str:
        """Display the extracted Output_Label entries in a readable format."""
        res = ""

        entries = (
            output_labels.labels
            if isinstance(output_labels, List_Output_Label)
            else output_labels
        )

        header = "| Entry | Activity | Student Labeled Spaces | Student Labeled Subspaces | Results | Reason |\n"
        separator = "| ---: | --- | --- | --- | --- | --- |\n"
        print(header.strip())
        res += header
        print(separator.strip())
        res += separator
        for idx, entry in enumerate(entries, start=1):
            activity = entry.activity.replace("|", "\\|")
            spaces = ", ".join(entry.student_labeled_spaces).replace("|", "\\|")
            subspaces = ", ".join(entry.student_labeled_subspaces).replace("|", "\\|")
            reason = entry.Reason.replace("|", "\\|")
            results_str = ", ".join(str(v) for v in entry.result)
            row = (
                f"| {idx} | {activity} | {spaces} | {subspaces} | "
                f"{results_str} | {reason} |\n"
            )
            print(row.strip())
            res += row
        return res


if __name__ == "__main__":
    from core.preprocessing import PreProcessor
    from core.processing import Processing

    final_processor = FinalProcessing()
    processor = Processing()
    preprocessor = PreProcessor()
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "progress_report_1.pdf",
    )
    # Preprocess to get student labeled data
    student_table_data = preprocessor.invoke(pdf_path)
    print("Student Labeled Data:")
    preprocessor.display_list_data_table(student_table_data)

    # Process to get LLM labeled data
    llm_table_data = processor.classify_table(student_table_data)
    print("\nLLM Labeled Data:")
    processor.display_list_data_table(llm_table_data)

    # Final evaluation
    final_output = final_processor.final_eval(student_table_data, llm_table_data)
    print("\nFinal Evaluation Results:")
    final_processor.display_output_labels(final_output)
