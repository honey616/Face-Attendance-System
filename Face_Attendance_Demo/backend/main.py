from pathlib import Path
import sys
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# BACKEND PATH
# ============================================================

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


# ============================================================
# FACE ATTENDANCE ENGINE
# ============================================================

from attendance_engine_source import (
    FaceProcessingError,
    generate_embedding,
    load_employees,
    process_captured_frame,
    recognize_employee,
    register_employee,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Employee Face Attendance API",
    description="Face recognition based employee attendance backend",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "employee-attendance-api",
    }


# ============================================================
# GET EMPLOYEES
# ============================================================

@app.get("/employees")
def employees() -> Dict[str, Any]:

    records = load_employees()

    return {
        "count": len(records),
        "employees": [
            {
                "employee_id": employee_id,
                "name": str(
                    record.get("name") or employee_id
                ),
            }
            for employee_id, record in sorted(records.items())
        ],
    }


# ============================================================
# EMPLOYEE REGISTRATION
# ============================================================

@app.post("/employees/register")
async def register_employee_api(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    front: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    employee_id = employee_id.strip().upper()
    employee_name = " ".join(
        employee_name.strip().split()
    )

    if not employee_id:
        raise HTTPException(
            status_code=400,
            detail="Employee ID is required.",
        )

    if not employee_name:
        raise HTTPException(
            status_code=400,
            detail="Employee name is required.",
        )

    # --------------------------------------------------------
    # Check duplicate employee ID
    # --------------------------------------------------------

    employees = load_employees()

    if employee_id in employees:
        raise HTTPException(
            status_code=409,
            detail=f"Employee {employee_id} is already registered.",
        )

    # --------------------------------------------------------
    # Uploaded face images
    # --------------------------------------------------------

    uploaded_files = {
        "front": front,
        "left": left,
        "right": right,
    }

    images: Dict[str, np.ndarray] = {}
    embeddings: Dict[str, list] = {}

    try:

        # ----------------------------------------------------
        # Process all three face images
        # ----------------------------------------------------

        for view, upload in uploaded_files.items():

            data = await upload.read()

            if not data:
                raise HTTPException(
                    status_code=400,
                    detail=f"{view.capitalize()} image is empty.",
                )

            # ------------------------------------------------
            # Decode image
            # ------------------------------------------------

            image = cv2.imdecode(
                np.frombuffer(
                    data,
                    dtype=np.uint8,
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {view} face image.",
                )

            # ------------------------------------------------
            # Generate face embedding
            # ------------------------------------------------

            try:

                embedding = generate_embedding(
                    image
                )

            except FaceProcessingError as exc:

                raise HTTPException(
                    status_code=422,
                    detail={
                        "view": view,
                        "status": exc.code,
                        "message": exc.message,
                    },
                )

            except Exception as exc:

                raise HTTPException(
                    status_code=422,
                    detail={
                        "view": view,
                        "status": "face_processing_error",
                        "message": str(exc),
                    },
                )

            # ------------------------------------------------
            # Store image
            # ------------------------------------------------

            images[view] = image

            # ------------------------------------------------
            # Convert NumPy embedding to Python list
            # ------------------------------------------------

            embeddings[view] = np.asarray(
                embedding,
                dtype=np.float32,
            ).tolist()

        # ----------------------------------------------------
        # Save employee profile
        # ----------------------------------------------------

        register_employee(
            employee_id=employee_id,
            employee_name=employee_name,
            images=images,
            embeddings=embeddings,
        )

    except HTTPException:
        raise

    except OSError as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not save employee profile: {exc}",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Employee registration failed: {exc}",
        )

    # --------------------------------------------------------
    # Success response
    # --------------------------------------------------------

    return {
        "status": "registered",
        "message": "Employee registered successfully.",
        "employee": {
            "employee_id": employee_id,
            "name": employee_name,
        },
        "face_references": [
            "front",
            "left",
            "right",
        ],
    }


# ============================================================
# ATTENDANCE VERIFY
# ============================================================

@app.post("/attendance/verify")
async def verify_attendance(
    file: UploadFile = File(...),
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty image.",
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image = cv2.imdecode(
        np.frombuffer(
            data,
            dtype=np.uint8,
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image.",
        )

    # --------------------------------------------------------
    # Load employees
    # --------------------------------------------------------

    employees = load_employees()

    if not employees:

        return {
            "status": "no_employees",
            "message": "No employees are registered yet.",
        }

    # --------------------------------------------------------
    # Face recognition
    # --------------------------------------------------------

    try:

        recognition = recognize_employee(
            image,
            employees,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Face verification failed: {exc}",
        )

    # --------------------------------------------------------
    # Face not matched
    # --------------------------------------------------------

    if recognition.status != "matched":

        return {
            "status": recognition.status,
            "message": recognition.message,
        }

    # --------------------------------------------------------
    # Record attendance
    # --------------------------------------------------------

    try:

        outcome = process_captured_frame(
            image
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Attendance processing failed: {exc}",
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "status": outcome.get(
            "kind",
            "error",
        ),
        "message": outcome.get(
            "message",
            "",
        ),
        "captured": bool(
            outcome.get(
                "captured",
                False,
            )
        ),
    }