# SIIP HCD Classifier API Documentation

This API provides services for classifying Human-Centered Design (HCD) activities from PDF documents and managing manual labeling of activities.

**Base URL**: `http://<host>:<port>` (e.g., `http://localhost:8000`)

---

## endpoints

### 1. Health Check
Checks if the API is running.

- **URL**: `/health`
- **Method**: `GET`
- **Auth**: None
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Root Info
Returns basic API information and available endpoints.

- **URL**: `/`
- **Method**: `GET`
- **Auth**: None
- **Response**:
  ```json
  {
    "message": "SIIP HCD Classifier API",
    "endpoints": {
      "health": "/health",
      "classify": "/classify",
      "docs": "/docs"
    }
  }
  ```

### 3. Classify PDF
Upload a PDF file to extract and classify HCD activities. This is the main core feature.

- **URL**: `/classify`
- **Method**: `POST`
- **Auth**: None
- **Content-Type**: `multipart/form-data`
- **Request Body**:
  - `file`: The PDF file to be classified (binary).
- **Response**: `ClassificationResponse` object.

  ```json
  {
    "student_labels": {
      "tables": [
        {
          "activity": "Interviewed users about their needs",
          "HCD_Spaces": ["understand"],
          "HCD_Subspaces": ["empathize"]
        }
      ]
    },
    "space_summary": {
      "understand": {
        "score": 1,
        "present": ["empathize"],
        "not_present": ["explore", "observe", "reflect"]
      },
      "synthesize": {
        "score": -1,
        "present": [],
        "not_present": ["debrief", "organize", "interpret", "define"]
      },
      "ideate": {
        "score": -1,
        "present": [],
        "not_present": ["brainstorm", "propose", "plan", "narrow concepts"]
      },
      "prototype": {
        "score": -1,
        "present": [],
        "not_present": ["create", "engage", "evaluate", "iterate"]
      },
      "implement": {
        "score": -1,
        "present": [],
        "not_present": ["support", "sustain", "evolve", "execute"]
      }
    }
  }
  ```

The summary includes all 20 rubric subspaces. A subspace with multiple
student-label evaluations is marked `passed` if any evaluation is `1`;
otherwise, it is `not_enough_evidence` if any evaluation is `0`; otherwise it
is `incorrect`. Subspaces without a student evaluation are assessed against
the report by the LLM: `1` means clear evidence was found, and `0` means the
evidence is absent or unclear. The output does not identify inferred
subspaces or include inference feedback. Each space's `score` is `1` if any
of its four subspaces passes, `-1` if all four are incorrect, and `0`
otherwise. The `final_labels` response omits the evaluator's `Reason` text.
The LLM classification is still used internally to produce `final_labels`
and the summary, but its intermediate labels are not included in the response.

### 4. Fetch Unlabeled Activity
Retrieves a single unlabeled activity from the database for manual labeling.

- **URL**: `/fetch-unlabeled`
- **Method**: `GET`
- **Auth**: None
- **Response**:
  ```json
  {
    "rowid": 123,
    "Activity": "Conducted a survey on campus."
  }
  ```
  *Returns `{"rowid": null, "Activity": null}` if no unlabeled activities are found.*

### 5. Label Activity
Submit a manual label for a specific activity.

- **URL**: `/label-activity`
- **Method**: `POST`
- **Auth**: None
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "rowid": 123,
    "HCD_Space": "Observation",
    "HCD_Subspace": "User Research",
    "Reason": "Explicit mention of survey.",
    "Annotator": "jdoe"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Activity 123 labeled successfully"
  }
  ```

### 6. Label Statistics
Returns statistics about activity labeling progress, including total activities, number of labeled activities, and number of unlabeled activities.

- **URL**: `/label-stats`
- **Method**: `GET`
- **Auth**: None
- **Response**:
  ```json
  {
    "total": 100,
    "labeled": 45,
    "unlabeled": 55
  }
  ```

---

## Data Models

### Student_HCD_Label
Structured representation of an activity extracted from the PDF.
| Field | Type | Description |
|---|---|---|
| `activity` | string | Description of the activity. |
| `HCD_Spaces` | list[string] | List of HCD spaces identified. |
| `HCD_Subspaces` | list[string] | List of HCD subspaces identified. |

### Output_Label
Final evaluation of the classification.
| Field | Type | Description |
|---|---|---|
| `activity` | string | Original activity description. |
| `student_labeled_spaces` | list[string] | Normalized spaces from the student. |
| `student_labeled_subspaces` | list[string] | Normalized subspaces from the student. |
| `result` | list[int] | Evaluation result per subspace (1: Correct, 0: Not enough evidence, -1: Incorrect). |
| `Reason` | string | Explanation for the evaluation result. |
