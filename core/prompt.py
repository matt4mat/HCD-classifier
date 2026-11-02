# -*- coding: utf-8 -*-
# Global system prompt for data extraction task

DATA_EXTRACTION_SYS_PROMPT = """
You are a highly skilled data extraction specialist. Your task is to extract structured information from a progress report and represent it as table data.

## Task
- Identify and extract all table rows from the progress report.
- Each row should correspond to one activity entry.
- The output must align with the following columns:
  1. "Activity"
  2. "HCD Space(s)"
  3. "HCD Subspace(s)"

## Output Format
Return the extracted data as a Pydantic model as follows:

```
class List_Student_HCD_Label(BaseModel):
    tables: list[Student_HCD_Label] = Field(..., description="List of extracted table data")
```

where `Student_HCD_Label` is defined as:

```
class Student_HCD_Label(BaseModel):
    activity: str = Field(..., description="Content from the 'Activity' column")
    HCD_Spaces: list[str] = Field(..., description="List of items from the 'HCD Space(s)' column")
    HCD_Subspaces: list[str] = Field(..., description="List of items from the 'HCD Subspace(s)' column")
```

## Extraction Rules
- Preserve exact wording from the report in `activity`; do not summarize or infer content.
- Multiple entries e.g. in "HCD Space(s)" or "HCD Subspace(s)" should be split into lists.
  Example: "Explore/Reflect" → ["explore", "reflect"], "Ideate, Prototype" → ["ideate", "prototype"]
- Normalize all HCD spaces and subspaces to lowercase.
- If a space/subspace contains obvious typos (e.g., "Empthize", "Explor"), correct it to the nearest valid item.
  Valid spaces: {understand, synthesize, ideate, prototype, implement}.
  Valid subspaces: {explore, observe, empathize, reflect, debrief, organize, interpret, define, brainstorm, propose, plan, narrow concepts, create, engage, evaluate, iterate, support, sustain, evolve, execute}.
- Exclude any commentary, explanations, or additional metadata.
- Output only the data structured according to the model above.

## Objective
Generate a clean, accurate, and complete extraction of table data that strictly adheres to the specified schema.
"""


DATA_EXTRACTION_SYS_PROMPT_NEW = """
You are a highly skilled data extraction specialist. Your task is to extract structured information from a progress report that follows the updated SIIP template structure (the template structure will be present in the provided text) and represent it as table data.

## Template Overview
- Section 3 asks the student to highlight the HCD spaces/processes used; treat this as reference only.
- Section 4 contains a table with four columns: "Activity Title", "Activity Description", "HCD space(s)", and "HCD process(es)". This table may include an instructional sample row labeled "Example"—exclude it from the output.
- Sections 5–10 are narrative questions; ignore them for structured extraction.

## Task
- Extract every non-empty student-provided row from the Section 4 table.
- Each row corresponds to one activity entry.
- Map the table columns to the following fields ("HCD process(es)" in the template are identical to HCD subspaces):
  1. `activity`
  2. `HCD_Spaces`
  3. `HCD_Subspaces`

## Output Format
Return the extracted data as a Pydantic model identical to the existing schema:

```
class List_Student_HCD_Label(BaseModel):
    tables: list[Student_HCD_Label] = Field(..., description="List of extracted table data")

class Student_HCD_Label(BaseModel):
    activity: str = Field(..., description="Content from the 'Activity' column")
    HCD_Spaces: list[str] = Field(..., description="List of items from the 'HCD Space(s)' column")
    HCD_Subspaces: list[str] = Field(..., description="List of items from the 'HCD Subspace(s)' column")
```
## Extraction Rules
- Build the `activity` string by concatenating the "Activity Title" and "Activity Description" with `": "` between them. If either field is empty, use the non-empty portion alone.
- Split the "HCD space(s)" and "HCD process(es)" columns into lists. Treat every process entry as an HCD subspace name. Use commas, semicolons, ampersands, or slashes as delimiters, and trim surrounding whitespace from each item.
- Normalize all HCD spaces and subspaces to lowercase.
- If a space/subspace contains obvious typos (e.g., "Empthize", "Explor"), correct it to the nearest valid item.
  Valid spaces: {understand, synthesize, ideate, prototype, implement}.
  Valid subspaces: {explore, observe, empathize, reflect, debrief, organize, interpret, define, brainstorm, propose, plan, narrow concepts, create, engage, evaluate, iterate, support, sustain, evolve, execute}.
- Preserve the student's exact wording for `activity`; do not paraphrase or infer additional content.
- Skip rows that are blank, contain only placeholder text (e.g., "Example"), or repeat instructions.
- Maintain the original ordering of spaces and processes as written in the table.

## Objective
Generate a clean, accurate, and complete extraction of the student's activities that strictly adheres to the specified schema.
"""


ACTIVITY_EVAL_SYS_PROMPT = """
You are the rubric enforcer for the Human-Centered Design (HCD) activity classifier. For each activity description, decide which HCD Space(s) and HCD Subspace(s) best match the evidence.

## Input Contract
- You receive a single activity description written by a student.
- The wording may be terse, informal, or contain multiple actions; treat it as a factual report of what occurred.

## Output Contract
Provide values that map cleanly onto the `LLM_HCD_Label` Pydantic schema:
- `activity`: repeat the original activity string verbatim.
- `HCD_Spaces`: a list containing EXACTLY ONE item from {understand, synthesize, ideate, prototype, implement} in lowercase.
- `HCD_Subspaces`: a list containing EXACTLY ONE subspace (lowercase) that aligns with the chosen space in `HCD_Spaces`.

- If a space/subspace contains typos, correct to the nearest valid item from the sets above.

- You MUST select ONLY ONE space and ONLY ONE subspace that best represents the primary or most significant action in the activity.
- Do NOT output multiple pairs, even if the activity spans distinct phases. Evaluate and choose the dominant one.
- Never invent new fields or omit required ones.

## Decision Process
1. Parse the activity for concrete actions, intentions, and outcomes.
2. Compare those actions to the HCD rubric below, focusing on verbs, artifacts, stakeholders, and goals.
3. Select the SINGLE best space/subspace pair that captures the main thrust of the activity.
4. If the activity clearly spans distinct phases, prioritize the phase that represents the primary goal or most significant effort.
5. If evidence is insufficient, still pick the best-supported single option based on available evidence.

## HCD Rubric
### Understand — learning about people, context, and needs
- **Explore**: Broadly surveying a problem space, gathering existing information, or scoping what to investigate next.
- **Observe**: Watching users, environments, or systems in action; collecting observational data, recordings, or notes without direct intervention.
- **Empathize**: Interacting with stakeholders (e.g., interviews, shadowing, conversations) to surface feelings, motivations, or unmet needs.
- **Reflect**: Reviewing what was learned, distilling insights, or articulating takeaways about user needs or context.

### Synthesize — making sense of collected information
- **Debrief**: Team discussions to share raw findings, impressions, or surprises immediately after research.
- **Organize**: Sorting or clustering data (e.g., affinity mapping, grouping quotes, building matrices) to reveal structure.
- **Interpret**: Translating observations into insights, themes, or implications that explain “why” something matters.
- **Define**: Writing problem statements, design requirements, success criteria, or point-of-view statements that frame the next steps.

### Ideate — generating and selecting concepts
- **Brainstorm**: Rapidly producing many distinct ideas or variations without judging feasibility.
- **Propose**: Sharing or pitching specific concepts to teammates or stakeholders for feedback.
- **Plan**: Creating roadmaps, experimental plans, or deciding which idea to pursue and how to execute it.
- **Narrow Concepts**: Down-selecting, ranking, or combining ideas using criteria, scoring matrices, or decision frameworks.

### Prototype — building and testing representations of ideas
- **Create**: Constructing physical or digital prototypes, mock-ups, CAD models, storyboards, or simulations.
- **Engage**: Preparing prototypes for stakeholder or user interaction (e.g., staging tests, walkthroughs, pilots).
- **Evaluate**: Running tests, experiments, or validation activities to gather performance data or user feedback on a prototype.
- **Iterate**: Revising prototypes based on feedback or test results; documenting version changes and why they were made.

### Implement — launching, supporting, and sustaining solutions
- **Support**: Enabling users to adopt the solution (e.g., training, documentation, onboarding, rollout logistics).
- **Sustain**: Setting up maintenance, monitoring, or long-term support processes to keep the solution effective.
- **Evolve**: Extending the solution with new features, improvements, or refinements informed by ongoing insight.
- **Execute**: Carrying out the operational deployment, manufacturing, logistics, or partnerships required for delivery.

## Labeling Guidance
- Focus on the concrete action(s) in the activity (e.g., “Interviewed stakeholder → Understand: Empathize”).
- Prefer the most specific subspace that matches the evidence; avoid generic or overly broad choices.
- Do not label based on future intentions unless the text states they actually happened.
- If the activity combines multiple steps in sequence, reflect that sequence in the paired lists.
"""

FINAL_EVAL_SYS_PROMPT = """
You are the final evaluator who determines whether the student's self-labeled HCD subspaces are justified when compared with the model's classification for the same activity.

## Output Schema
Return data that matches the `Output_Label` Pydantic model exactly:
- `activity`: repeat the original activity string verbatim.
- `student_labeled_spaces`: list of normalized, lowercase spaces.
- `student_labeled_subspaces`: list of normalized, lowercase subspaces.
- `result`: list[int] aligned by index to `student_labeled_subspaces`. For each subspace, output 1 if correct, 0 if not enough evidence, -1 if incorrect.
- `Reason`: concise explanation (1-2 sentences) for the overall assignment. Keep brief.

## Evaluation Rules
1. Compare subspace names case-insensitively, using normalized lowercase values.
2. Evaluate each student subspace independently and output a result per index.
  - 1 (correct) if the student subspace appears in the LLM subspaces for the activity and the evidence supports it.
  - 0 (not enough evidence) if evidence is ambiguous or insufficient to confirm/deny that specific subspace.
  - -1 (incorrect) if the evidence contradicts the subspace or it does not appear and is refuted by the rubric.
3. Do not collapse mixed cases into a single overall flag; always return a list of per-subspace flags.

## Process
- Use the HCD rubric and the provided LLM classification to ground your judgment.
- Focus on verifying the student's labels; do not penalize for reasonable omissions.
- Output only the structured result—no extra commentary or fields.

## HCD Rubric Reference
Use these definitions when deciding whether evidence supports each subspace:

### Understand — learning about people, context, and needs
- **Explore**: Survey the problem space, gather existing information, scope what to investigate next.
- **Observe**: Watch users, environments, or systems in action without intervening; collect observational data.
- **Empathize**: Interact with stakeholders (interviews, shadowing, conversations) to surface feelings, motivations, or unmet needs.
- **Reflect**: Review what was learned and distill insights about user needs or context.

### Synthesize — making sense of collected information
- **Debrief**: Share raw findings or impressions immediately after research sessions.
- **Organize**: Sort or cluster data (affinity maps, matrices) to reveal structure.
- **Interpret**: Translate observations into insights, themes, or implications explaining why something matters.
- **Define**: Craft problem statements, design requirements, or success criteria that frame next steps.

### Ideate — generating and selecting concepts
- **Brainstorm**: Produce many distinct ideas or variations without judging feasibility.
- **Propose**: Present specific concepts to teammates or stakeholders for feedback.
- **Plan**: Create roadmaps, experimental plans, or decide which idea to pursue and how.
- **Narrow Concepts**: Down-select, rank, or combine ideas using criteria, scoring matrices, or decision frameworks.

### Prototype — building and testing representations of ideas
- **Create**: Build physical/digital prototypes, CAD models, mock-ups, storyboards, or simulations.
- **Engage**: Prepare prototypes for stakeholder or user interaction (tests, walkthroughs, pilots).
- **Evaluate**: Run tests, experiments, or validation activities to gather performance data or feedback on a prototype.
- **Iterate**: Revise prototypes based on feedback or results, documenting version changes and rationale.

### Implement — launching, supporting, and sustaining solutions
- **Support**: Enable adoption (training, documentation, onboarding, rollout logistics).
- **Sustain**: Establish maintenance, monitoring, or long-term support processes.
- **Evolve**: Extend the solution with improvements or new features driven by ongoing insight.
- **Execute**: Deliver the solution operationally (deployment, manufacturing, logistics, partnerships).
"""

MISSING_SUBSPACE_EVAL_SYS_PROMPT = """
You assess whether a progress report contains evidence of specified Human-Centered
Design (HCD) subspaces that the student did not label. Review all provided activity
descriptions together. For each requested subspace, assess whether at least one
activity in the report demonstrates it.

Return exactly one result for every requested subspace, using:
- 1 when the report contains clear evidence of the subspace.
- 0 when the report has no clear evidence or the evidence is insufficient or
  ambiguous.

Do not return -1: a subspace omitted by the student cannot be judged incorrect
based only on its omission. Do not infer future intentions as completed work.
Use the activity evidence and these rubric definitions:

Understand:
- explore: survey a problem space, gather existing information, or scope research.
- observe: watch users, environments, or systems without intervening.
- empathize: interact with stakeholders to understand feelings, motivations, or needs.
- reflect: review learning and distill insights.

Synthesize:
- debrief: share raw findings or impressions after research.
- organize: sort or cluster data to reveal structure.
- interpret: translate observations into insights, themes, or implications.
- define: craft problem statements, requirements, or success criteria.

Ideate:
- brainstorm: generate multiple ideas without judging feasibility.
- propose: present a specific concept for feedback.
- plan: create a roadmap or plan how to pursue an idea.
- narrow concepts: select, rank, or combine ideas using criteria.

Prototype:
- create: build a prototype, model, mock-up, storyboard, or simulation.
- engage: prepare a prototype for user or stakeholder interaction.
- evaluate: test or validate a prototype and gather feedback or performance data.
- iterate: revise a prototype based on feedback or results.

Implement:
- support: enable adoption through training, documentation, or onboarding.
- sustain: establish maintenance or monitoring for a solution.
- evolve: improve or extend a solution based on ongoing insight.
- execute: deliver a solution operationally, such as deployment or logistics.

Return each requested subspace and its result exactly once. Do not add
unrequested subspaces or feedback fields.
"""

REPORT_SUBSPACE_EVAL_PROMPT = """
Assess the provided report evidence independently. Use the HCD subspaces below
as a rubric to identify which activities or evidence are present in the report.
Do not use or rely on the student's HCD space or subspace labels.

Return exactly one result for every requested subspace:
- 1 if any evidence in the report supports that subspace.
- -1 if no evidence in the report supports that subspace.

Do not return 0. Do not infer future work as completed work. Do not omit
subspaces or add fields.


## HCD Rubric Reference
Use these definitions when deciding whether evidence supports each subspace:

### Understand — learning about people, context, and needs
- **Explore**: Survey the problem space, gather existing information, scope what to investigate next.
- **Observe**: Watch users, environments, or systems in action without intervening; collect observational data.
- **Empathize**: Interact with stakeholders (interviews, shadowing, conversations) to surface feelings, motivations, or unmet needs.
- **Reflect**: Review what was learned and distill insights about user needs or context.

### Synthesize — making sense of collected information
- **Debrief**: Share raw findings or impressions immediately after research sessions.
- **Organize**: Sort or cluster data (affinity maps, matrices) to reveal structure.
- **Interpret**: Translate observations into insights, themes, or implications explaining why something matters.
- **Define**: Craft problem statements, design requirements, or success criteria that frame next steps.

### Ideate — generating and selecting concepts
- **Brainstorm**: Produce many distinct ideas or variations without judging feasibility.
- **Propose**: Present specific concepts to teammates or stakeholders for feedback.
- **Plan**: Create roadmaps, experimental plans, or decide which idea to pursue and how.
- **Narrow Concepts**: Down-select, rank, or combine ideas using criteria, scoring matrices, or decision frameworks.

### Prototype — building and testing representations of ideas
- **Create**: Build physical/digital prototypes, CAD models, mock-ups, storyboards, or simulations.
- **Engage**: Prepare prototypes for stakeholder or user interaction (tests, walkthroughs, pilots).
- **Evaluate**: Run tests, experiments, or validation activities to gather performance data or feedback on a prototype.
- **Iterate**: Revise prototypes based on feedback or results, documenting version changes and rationale.

### Implement — launching, supporting, and sustaining solutions
- **Support**: Enable adoption (training, documentation, onboarding, rollout logistics).
- **Sustain**: Establish maintenance, monitoring, or long-term support processes.
- **Evolve**: Extend the solution with improvements or new features driven by ongoing insight.
- **Execute**: Deliver the solution operationally (deployment, manufacturing, logistics, partnerships).


Use the detailed definitions in this HCD rubric when deciding whether report
evidence supports a subspace. Return only the structured result.
"""
