"""
Employee Face Attendance System
===============================

Streamlit + DeepFace (Facenet512 / retinaface / cosine) + streamlit-webrtc.

* Attendance: live camera, automatic capture after a 5 second countdown, no
  capture button, no overlays. DeepFace runs ONLY on the captured frame.
* Employee Registration: step-by-step front / left / right capture, one
  camera at a time.
* Attendance Records: filterable table with CSV export and a daily summary.

Local storage: employees.json, attendance.csv and the known_faces/ folder.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Keep the console quiet; must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from deepface import DeepFace  # noqa: E402
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer  # noqa: E402

# =============================================================================
# CONFIGURATION (change values here)
# =============================================================================
APP_TITLE = "Employee Face Attendance System"

# Face recognition
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"          # "cosine", "euclidean" or "euclidean_l2"
MATCH_THRESHOLD = 0.30              # a lower distance means a closer match
MIN_FACE_CONFIDENCE = 0.90          # ignore weak detections from the detector

# Attendance behaviour
SCAN_COOLDOWN_SECONDS = 60          # no second record within this many seconds
COUNTDOWN_SECONDS = 5               # countdown before the automatic capture
FACE_POLL_INTERVAL_SECONDS = 0.35   # how often the live frame is checked for faces
RESULT_DISPLAY_SECONDS = 2.5        # how long a result stays up before watching for the next employee

# Camera behaviour
CAMERA_READY_TIMEOUT_SECONDS = 20   # how long to wait for the first video frame
FRAME_STALE_SECONDS = 3.0           # a frame older than this means the camera stalled

# Timestamps are stored in this time zone. Streamlit Community Cloud runs in
# UTC, so keep this set for a deployed app. Use None for the server clock.
TIMEZONE: Optional[str] = "Asia/Kolkata"

# Storage
BASE_DIR = Path(__file__).resolve().parent
EMPLOYEES_FILE = BASE_DIR / "employees.json"
ATTENDANCE_FILE = BASE_DIR / "attendance.csv"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"

# =============================================================================
# CONSTANTS
# =============================================================================
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

EVENT_CHECK_IN = "CHECK-IN"
EVENT_CHECK_OUT = "CHECK-OUT"

ATTENDANCE_COLUMNS = ["Employee ID", "Employee Name", "Date", "Time", "Event"]
SUMMARY_COLUMNS = [
    "Employee ID",
    "Employee Name",
    "First Check-in",
    "Last Check-out",
    "Visits",
    "Working Time",
]

FACE_VIEWS = ("front", "left", "right")
VIEW_LABELS = {"front": "Front Face", "left": "Left Face", "right": "Right Face"}
VIEW_INSTRUCTIONS = {
    "front": "Look straight at the camera.",
    "left": "Turn your head slightly to your left.",
    "right": "Turn your head slightly to your right.",
}
STEP_VIEWS = {1: "front", 2: "left", 3: "right"}
REVIEW_STEP = 4

EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Z0-9_-]{1,32}$")

NAV_ATTENDANCE = "Attendance"
NAV_REGISTRATION = "Employee Registration"
NAV_RECORDS = "Attendance Records"
NAV_SECTIONS = [NAV_ATTENDANCE, NAV_REGISTRATION, NAV_RECORDS]

# Session-state keys
WEBRTC_KEY = "attendance_camera"
REG_STATE_KEY = "registration"
REG_EPOCH_KEY = "registration_epoch"
REG_NOTICE_KEY = "registration_notice"

# User-facing messages (kept short and free of technical details)
MSG_NO_FACE = "Waiting for employee..."
MSG_MULTIPLE_FACES = "Only one employee should be visible."
MSG_VERIFICATION_FAILED = "Employee verification failed."
MSG_VERIFICATION_ERROR = "Verification could not be completed. Please try again."
MSG_NO_EMPLOYEES = "No employees are registered yet. Please register an employee first."
MSG_CAMERA_TIMEOUT = (
    "Camera access has not been granted yet. Please allow camera access when your browser "
    "asks — this page connects automatically once access is granted."
)
MSG_CAMERA_LOST = "The camera connection was interrupted. Reconnecting automatically..."
MSG_SAVE_ATTENDANCE_FAILED = (
    "Attendance could not be saved. Please make sure attendance.csv is not open in "
    "another program and try again."
)

logger = logging.getLogger("face_attendance")


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def now_local() -> datetime:
    """Current time as a naive datetime in the configured time zone."""
    try:
        if TIMEZONE:
            return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)
    except Exception:  # unknown zone or missing tz database
        logger.warning("Time zone '%s' is unavailable; using server time.", TIMEZONE)
    return datetime.now()


def format_duration(total_seconds: float) -> str:
    """Format seconds as 'Xh YYm ZZs'."""
    total = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


@st.cache_resource
def get_storage_lock() -> threading.RLock:
    """One process-wide lock so parallel sessions cannot corrupt the files."""
    return threading.RLock()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file so a crash never leaves a half-written file."""
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _quarantine_corrupt_file(path: Path) -> None:
    """Keep an unreadable data file for inspection instead of overwriting it."""
    try:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        os.replace(path, path.with_name(f"{path.name}.corrupt-{stamp}"))
    except OSError:
        logger.exception("Could not move the unreadable file %s aside", path)


# =============================================================================
# STORAGE: EMPLOYEES
# =============================================================================
def load_employees() -> Dict[str, Dict[str, Any]]:
    """Load employees.json. Returns an empty dict if missing or unreadable."""
    with get_storage_lock():
        if not EMPLOYEES_FILE.exists():
            return {}
        try:
            raw = EMPLOYEES_FILE.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.exception("employees.json is not valid JSON")
            _quarantine_corrupt_file(EMPLOYEES_FILE)
            return {}
        except OSError:
            logger.exception("employees.json could not be read")
            return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def save_employees(employees: Dict[str, Dict[str, Any]]) -> None:
    """Save employees.json (raises OSError if the file cannot be written)."""
    with get_storage_lock():
        _atomic_write_text(EMPLOYEES_FILE, json.dumps(employees, indent=2))


def get_reference_embeddings(record: Dict[str, Any]) -> List[np.ndarray]:
    """All valid reference embeddings (front, left, right) of one employee."""
    raw = record.get("embeddings")
    if not isinstance(raw, list) or not raw:
        views = record.get("embedding_views")
        raw = list(views.values()) if isinstance(views, dict) else []
    embeddings: List[np.ndarray] = []
    for item in raw:
        try:
            array = np.asarray(item, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if array.ndim == 1 and array.size > 0:
            embeddings.append(array)
    return embeddings


# =============================================================================
# STORAGE: ATTENDANCE
# =============================================================================
def _empty_attendance() -> pd.DataFrame:
    return pd.DataFrame(columns=ATTENDANCE_COLUMNS)


def load_attendance() -> pd.DataFrame:
    """Load attendance.csv. Returns an empty table if missing or unreadable."""
    with get_storage_lock():
        if not ATTENDANCE_FILE.exists():
            return _empty_attendance()
        try:
            frame = pd.read_csv(ATTENDANCE_FILE, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            return _empty_attendance()  # zero-byte file
        except (pd.errors.ParserError, UnicodeDecodeError):
            logger.exception("attendance.csv could not be parsed")
            return _empty_attendance()
        except OSError:
            logger.exception("attendance.csv could not be read")
            return _empty_attendance()
    for column in ATTENDANCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[ATTENDANCE_COLUMNS].copy()


def save_attendance(attendance: pd.DataFrame) -> None:
    """Save attendance.csv (raises OSError if the file cannot be written)."""
    with get_storage_lock():
        temp_path = ATTENDANCE_FILE.with_name(ATTENDANCE_FILE.name + ".tmp")
        attendance[ATTENDANCE_COLUMNS].to_csv(temp_path, index=False)
        os.replace(temp_path, ATTENDANCE_FILE)


def ensure_storage() -> bool:
    """Create the data folder and empty data files when they do not exist."""
    try:
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        with get_storage_lock():
            if not EMPLOYEES_FILE.exists():
                _atomic_write_text(EMPLOYEES_FILE, "{}")
            if not ATTENDANCE_FILE.exists():
                save_attendance(_empty_attendance())
        return True
    except OSError:
        logger.exception("Storage could not be initialised")
        return False


# =============================================================================
# ATTENDANCE LOGIC
# =============================================================================
@dataclass
class AttendanceOutcome:
    status: str            # "recorded" or "cooldown"
    event: str
    timestamp: datetime


def _parse_timestamp(date_text: str, time_text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(f"{date_text} {time_text}", f"{DATE_FORMAT} {TIME_FORMAT}")
    except (TypeError, ValueError):
        return None


def _events_for_day(
    attendance: pd.DataFrame, employee_id: str, date_text: str
) -> pd.DataFrame:
    """One employee's events on one date, oldest first."""
    mask = (attendance["Employee ID"] == employee_id) & (attendance["Date"] == date_text)
    return attendance.loc[mask].sort_values("Time", kind="stable")


def determine_next_event(last_event: Optional[str]) -> str:
    """nothing -> CHECK-IN, CHECK-IN -> CHECK-OUT, CHECK-OUT -> CHECK-IN."""
    return EVENT_CHECK_OUT if last_event == EVENT_CHECK_IN else EVENT_CHECK_IN


def record_attendance(
    employee_id: str, employee_name: str, now: Optional[datetime] = None
) -> AttendanceOutcome:
    """
    Record the next alternating event for the employee.

    Inside the cooldown window nothing is written. The whole decision is made
    under one lock so two simultaneous scans cannot create duplicate records.
    """
    now = now or now_local()
    today = now.strftime(DATE_FORMAT)
    with get_storage_lock():
        attendance = load_attendance()
        todays_events = _events_for_day(attendance, employee_id, today)

        last_event: Optional[str] = None
        if not todays_events.empty:
            last_row = todays_events.iloc[-1]
            last_event = str(last_row["Event"])
            last_time = _parse_timestamp(today, str(last_row["Time"]))
            if last_time is not None:
                elapsed = (now - last_time).total_seconds()
                if 0 <= elapsed < SCAN_COOLDOWN_SECONDS:
                    return AttendanceOutcome("cooldown", last_event, last_time)

        event = determine_next_event(last_event)
        new_row = pd.DataFrame(
            [
                {
                    "Employee ID": employee_id,
                    "Employee Name": employee_name,
                    "Date": today,
                    "Time": now.strftime(TIME_FORMAT),
                    "Event": event,
                }
            ]
        )
        save_attendance(pd.concat([attendance, new_row], ignore_index=True))
    return AttendanceOutcome("recorded", event, now)


def calculate_daily_summary(
    attendance: pd.DataFrame, date_text: str, employee_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Per employee: first check-in, last check-out, number of visits (check-ins)
    and total working time from completed CHECK-IN -> CHECK-OUT pairs.
    """
    if attendance.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    day = attendance[attendance["Date"] == date_text]
    if employee_id:
        day = day[day["Employee ID"] == employee_id]

    rows: List[Dict[str, Any]] = []
    for emp_id, group in day.groupby("Employee ID", sort=True):
        events = []
        for _, record in group.iterrows():
            stamp = _parse_timestamp(str(record["Date"]), str(record["Time"]))
            if stamp is not None:
                events.append((stamp, str(record["Event"])))
        if not events:
            continue
        events.sort(key=lambda item: item[0])

        first_in: Optional[datetime] = None
        last_out: Optional[datetime] = None
        open_in: Optional[datetime] = None
        visits = 0
        worked = timedelta(0)
        for stamp, event in events:
            if event == EVENT_CHECK_IN:
                visits += 1
                if first_in is None:
                    first_in = stamp
                if open_in is None:
                    open_in = stamp
            elif event == EVENT_CHECK_OUT:
                last_out = stamp
                if open_in is not None:
                    worked += stamp - open_in
                    open_in = None

        rows.append(
            {
                "Employee ID": emp_id,
                "Employee Name": str(group["Employee Name"].iloc[-1]),
                "First Check-in": first_in.strftime(TIME_FORMAT) if first_in else "-",
                "Last Check-out": last_out.strftime(TIME_FORMAT) if last_out else "-",
                "Visits": visits,
                "Working Time": format_duration(worked.total_seconds()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compute_dashboard_metrics(
    employees: Dict[str, Dict[str, Any]], attendance: pd.DataFrame
) -> Dict[str, int]:
    """Numbers shown in the dashboard header."""
    today = now_local().strftime(DATE_FORMAT)
    todays = attendance[attendance["Date"] == today] if not attendance.empty else attendance
    checked_in = 0
    if not todays.empty:
        for _, group in todays.groupby("Employee ID"):
            latest = group.sort_values("Time", kind="stable").iloc[-1]["Event"]
            if latest == EVENT_CHECK_IN:
                checked_in += 1
    return {
        "registered": len(employees),
        "events_today": int(len(todays)),
        "checked_in": checked_in,
    }


# =============================================================================
# FACE RECOGNITION (DeepFace runs only when these functions are called)
# =============================================================================
class FaceProcessingError(Exception):
    """A face problem that can be shown to the user."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code            # no_face, multiple_faces, invalid_image, error
        self.message = message


@dataclass
class RecognitionResult:
    status: str                     # matched, no_match, no_employees, or a face error code
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    message: str = ""


@st.cache_resource(
    show_spinner="Preparing the face recognition engine. The first start can take a few minutes."
)
def warm_up_face_engine() -> bool:
    """Load the recognition and detection models once, before the first scan."""
    try:
        DeepFace.build_model(model_name=MODEL_NAME, task="facial_recognition")
        DeepFace.build_model(model_name=DETECTOR_BACKEND, task="face_detector")
        return True
    except Exception:
        logger.exception("Model warm-up failed; models will load on first use")
        return False


def _face_confidence(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("face_confidence", 1.0))
    except (TypeError, ValueError):
        return 0.0


def generate_embedding(image_bgr: np.ndarray) -> List[float]:
    """
    Return the embedding of the single face in the image (BGR array).

    Raises FaceProcessingError for no face, several faces or any other failure.
    """
    if (
        image_bgr is None
        or not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.size == 0
    ):
        raise FaceProcessingError("invalid_image", MSG_VERIFICATION_ERROR)

    try:
        results = DeepFace.represent(
            img_path=image_bgr,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        text = str(exc).lower()
        if "face" in text and "detect" in text:
            raise FaceProcessingError("no_face", MSG_NO_FACE) from exc
        logger.exception("DeepFace rejected the image")
        raise FaceProcessingError("error", MSG_VERIFICATION_ERROR) from exc
    except Exception as exc:
        logger.exception("DeepFace embedding failed")
        raise FaceProcessingError("error", MSG_VERIFICATION_ERROR) from exc

    if isinstance(results, dict):
        results = [results]
    faces = [item for item in results if _face_confidence(item) >= MIN_FACE_CONFIDENCE]
    if not faces:
        raise FaceProcessingError("no_face", MSG_NO_FACE)
    if len(faces) > 1:
        raise FaceProcessingError("multiple_faces", MSG_MULTIPLE_FACES)
    return [float(value) for value in faces[0]["embedding"]]


def count_valid_faces(frame_bgr: Optional[np.ndarray]) -> int:
    """
    Count faces in a live camera frame using the existing detector backend.

    This is detection only (no embeddings, no identity lookup), so it is
    cheap enough to run on every polled frame while watching for a single
    employee to step in front of the camera. It never calls the recognition
    model and is not a second recognition system.
    """
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return 0
    try:
        faces = DeepFace.extract_faces(
            img_path=frame_bgr,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=False,
        )
    except Exception:
        logger.exception("Live face detection failed")
        return 0
    count = 0
    for face in faces:
        try:
            confidence = float(face.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= MIN_FACE_CONFIDENCE:
            count += 1
    return count


def compute_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Distance between two embeddings using DISTANCE_METRIC."""
    if DISTANCE_METRIC == "cosine":
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0.0:
            return 1.0
        return float(1.0 - np.dot(first, second) / denominator)
    if DISTANCE_METRIC == "euclidean":
        return float(np.linalg.norm(first - second))
    if DISTANCE_METRIC == "euclidean_l2":
        first_norm = np.linalg.norm(first) or 1.0
        second_norm = np.linalg.norm(second) or 1.0
        return float(np.linalg.norm(first / first_norm - second / second_norm))
    raise ValueError(f"Unsupported distance metric: {DISTANCE_METRIC}")


def recognize_employee(
    image_bgr: np.ndarray, employees: Dict[str, Dict[str, Any]]
) -> RecognitionResult:
    """
    Compare the captured face with every reference embedding (front, left,
    right) of every employee. The closest single reference decides the match.
    """
    if not employees:
        return RecognitionResult("no_employees", message=MSG_NO_EMPLOYEES)

    try:
        query = np.asarray(generate_embedding(image_bgr), dtype=np.float32)
    except FaceProcessingError as exc:
        return RecognitionResult(exc.code, message=exc.message)

    best_id: Optional[str] = None
    best_distance = float("inf")
    for employee_id, record in employees.items():
        for reference in get_reference_embeddings(record):
            if reference.shape != query.shape:
                continue  # produced by a different model; cannot be compared
            distance = compute_distance(query, reference)
            if distance < best_distance:
                best_distance, best_id = distance, employee_id

    # The distance is logged on the server only and is never shown to the user.
    logger.info("Best match: %s (distance %.4f)", best_id, best_distance)

    if best_id is None or best_distance > MATCH_THRESHOLD:
        return RecognitionResult("no_match", message=MSG_VERIFICATION_FAILED)
    name = str(employees[best_id].get("name") or best_id)
    return RecognitionResult("matched", employee_id=best_id, employee_name=name)


def decode_image(data: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode image bytes to a BGR array; None if the data is not an image."""
    if not data:
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


# =============================================================================
# EMPLOYEE PROFILE CREATION
# =============================================================================
def save_reference_image(employee_id: str, view: str, image_bgr: np.ndarray) -> str:
    """Save one reference photo into known_faces/ and return its file name."""
    filename = f"{employee_id}_{view}.jpg"
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError("The image could not be encoded.")
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    (KNOWN_FACES_DIR / filename).write_bytes(buffer.tobytes())
    return filename


def register_employee(
    employee_id: str,
    employee_name: str,
    images: Dict[str, np.ndarray],
    embeddings: Dict[str, List[float]],
) -> None:
    """Save the reference photos and the employee record (raises OSError)."""
    with get_storage_lock():
        reference_images = {
            view: save_reference_image(employee_id, view, images[view]) for view in FACE_VIEWS
        }
        employees = load_employees()
        employees[employee_id] = {
            "name": employee_name,
            "embeddings": [embeddings[view] for view in FACE_VIEWS],
            "embedding_views": {view: embeddings[view] for view in FACE_VIEWS},
            "reference_images": reference_images,
            "registered_at": now_local().isoformat(timespec="seconds"),
        }
        save_employees(employees)


# =============================================================================
# WEBRTC (lightweight: only keeps the latest frame)
# =============================================================================
class LatestFrameProcessor(VideoProcessorBase):
    """
    Stores the newest camera frame and returns it untouched.

    It does not detect faces, draw anything or call DeepFace, so the live
    video stays clean and the processor stays lightweight.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._frame = None
        self._last_frame_at = 0.0

    def recv(self, frame):  # av.VideoFrame -> av.VideoFrame
        with self._lock:
            self._frame = frame
            self._last_frame_at = time.monotonic()
        return frame

    def has_recent_frame(self, max_age: float) -> bool:
        with self._lock:
            return self._frame is not None and (time.monotonic() - self._last_frame_at) <= max_age

    def get_latest_frame_bgr(self) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        try:
            return frame.to_ndarray(format="bgr24")
        except Exception:
            logger.exception("The camera frame could not be converted")
            return None


LOCAL_HOST_PATTERN = re.compile(r"^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$", re.IGNORECASE)
PUBLIC_STUN_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
]


def _is_local_session() -> bool:
    """True when the browser reaches this app through localhost."""
    if os.environ.get("WEBRTC_FORCE_STUN") == "1":
        return False
    try:
        host = str(st.context.headers.get("Host", ""))
    except Exception:
        return True
    return host == "" or bool(LOCAL_HOST_PATTERN.match(host))


def get_rtc_configuration() -> Dict[str, Any]:
    """
    ICE configuration for WebRTC.

    * localhost: no ICE servers, so nothing depends on a public STUN server.
    * anywhere else (for example Streamlit Community Cloud): public STUN, or
      your own STUN/TURN servers from the WEBRTC_ICE_SERVERS environment
      variable / secret (a JSON list such as [{"urls": ["turn:..."],
      "username": "...", "credential": "..."}]).
    """
    if _is_local_session():
        # Use the same working STUN configuration as the standalone camera test.
        # The browser still requests the local camera directly; STUN only helps
        # the WebRTC peer connection establish reliably.
        return {"iceServers": PUBLIC_STUN_SERVERS}
    custom = os.environ.get("WEBRTC_ICE_SERVERS", "").strip()
    if custom:
        try:
            servers = json.loads(custom)
            if isinstance(servers, list) and servers:
                return {"iceServers": servers}
        except json.JSONDecodeError:
            logger.warning("WEBRTC_ICE_SERVERS is not valid JSON; using public STUN.")
    return {"iceServers": PUBLIC_STUN_SERVERS}


# =============================================================================
# UI HELPERS
# =============================================================================
STYLES = """
<style>
:root {
    --bg: #0b1220;
    --panel: #111a2e;
    --panel-border: #1f2b47;
    --text: #e6eaf2;
    --muted: #93a0b8;
    --accent: #3b82f6;
    --ok: #22c55e;
    --warn: #f59e0b;
    --err: #ef4444;
}
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1150px; }
h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

.app-header { margin-bottom: 1.2rem; }
.app-title { font-size: 1.9rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.2px; }
.app-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 2px; }

[data-testid="stMetric"] {
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] p { color: var(--muted); }
[data-testid="stMetricValue"] { color: #f8fafc; }

.stButton > button, .stDownloadButton > button {
    background: var(--panel); color: var(--text);
    border: 1px solid var(--panel-border); border-radius: 8px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); color: #ffffff;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent); border-color: var(--accent); color: #ffffff;
}
.stButton > button:disabled { opacity: 0.45; }

.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: var(--panel); color: var(--text); border-color: var(--panel-border);
}

div[role="radiogroup"] > label {
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 8px; padding: 6px 16px; margin-right: 8px;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: #1d4ed8; border-color: var(--accent);
}

.status-card {
    border-radius: 10px; padding: 14px 18px; margin: 10px 0;
    border: 1px solid var(--panel-border); background: var(--panel);
    font-size: 1rem; line-height: 1.4;
}
.status-success { border-left: 4px solid var(--ok); }
.status-error { border-left: 4px solid var(--err); }
.status-warning { border-left: 4px solid var(--warn); }
.status-info { border-left: 4px solid var(--accent); }

.countdown-card {
    text-align: center; background: var(--panel);
    border: 1px solid var(--panel-border); border-radius: 12px;
    padding: 18px 12px; margin: 10px 0;
}
.countdown-caption { color: var(--muted); font-size: 0.95rem; }
.countdown-number { font-size: 5rem; font-weight: 700; color: #f8fafc; line-height: 1.1; }
</style>
"""


def inject_styles() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)


def message_html(kind: str, text: str) -> str:
    return f'<div class="status-card status-{kind}">{html.escape(text)}</div>'


def render_message(kind: str, text: str, target: Any = None) -> None:
    """Show a status message. kind: success, error, warning or info."""
    (target if target is not None else st).markdown(message_html(kind, text), unsafe_allow_html=True)


def render_countdown(placeholder: Any, remaining: int) -> None:
    placeholder.markdown(
        '<div class="countdown-card">'
        '<div class="countdown-caption">Please look at the camera. Capturing in</div>'
        f'<div class="countdown-number">{int(remaining)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard(placeholder: Any) -> None:
    metrics = compute_dashboard_metrics(load_employees(), load_attendance())
    with placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registered Employees", metrics["registered"])
        col2.metric("Today's Events", metrics["events_today"])
        col3.metric("Employees Checked In", metrics["checked_in"])
        col4.metric("System Status", "Online")


# =============================================================================
# ATTENDANCE PAGE
# =============================================================================
def reset_attendance_session() -> None:
    """Drop the camera session when the user leaves the Attendance section."""
    st.session_state.pop(WEBRTC_KEY, None)
    st.session_state.pop("attendance_webrtc_ctx", None)
    _reset_attendance_monitor_state()


def make_outcome(
    kind: str, message: str, captured: bool = False, summary: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    return {"kind": kind, "message": message, "captured": captured, "summary": summary}


def render_outcome(placeholder: Any, outcome: Dict[str, Any]) -> None:
    """Show a result message and, if present, the updated daily summary."""
    with placeholder.container():
        render_message(outcome["kind"], outcome["message"])
        summary = outcome.get("summary")
        if summary is not None and not summary.empty:
            st.markdown("**Today's Summary**")
            st.dataframe(summary, hide_index=True, use_container_width=True)


def process_captured_frame(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Run recognition on the captured frame and record attendance."""
    employees = load_employees()
    result = recognize_employee(frame_bgr, employees)

    if result.status == "matched" and result.employee_id and result.employee_name:
        try:
            outcome = record_attendance(result.employee_id, result.employee_name)
        except OSError:
            logger.exception("Attendance could not be saved")
            return make_outcome("error", MSG_SAVE_ATTENDANCE_FAILED, captured=True)

        summary = calculate_daily_summary(
            load_attendance(), now_local().strftime(DATE_FORMAT), result.employee_id
        )
        clock = outcome.timestamp.strftime(TIME_FORMAT)
        if outcome.status == "recorded":
            return make_outcome(
                "success",
                f"{outcome.event} recorded for {result.employee_name} at {clock}.",
                captured=True,
                summary=summary,
            )
        return make_outcome(
            "info",
            f"{result.employee_name}, your {outcome.event} was already recorded at {clock}. "
            "Please wait a moment before scanning again.",
            captured=True,
            summary=summary,
        )

    kind = {
        "no_face": "warning",
        "multiple_faces": "warning",
        "no_employees": "info",
        "no_match": "error",
    }.get(result.status, "error")
    return make_outcome(kind, result.message or MSG_VERIFICATION_ERROR, captured=True)


def _attendance_state_defaults() -> None:
    """Initialize state used by the non-blocking attendance monitor."""
    st.session_state.setdefault("attendance_countdown_deadline", None)
    st.session_state.setdefault("attendance_hold_until", 0.0)
    st.session_state.setdefault("attendance_stale_since", None)
    st.session_state.setdefault("attendance_ever_connected", False)


def _reset_attendance_monitor_state() -> None:
    st.session_state["attendance_countdown_deadline"] = None
    st.session_state["attendance_hold_until"] = 0.0
    st.session_state["attendance_stale_since"] = None
    st.session_state["attendance_ever_connected"] = False


@st.fragment(run_every=0.35)
def _attendance_monitor_fragment(
    dashboard_placeholder: Any,
    camera_status_placeholder: Any,
    detection_placeholder: Any,
) -> None:
    """
    Poll one camera frame per fragment run without blocking Streamlit's main
    script.  This is important for WebRTC: the component must be allowed to
    finish its browser-side signalling/permission handshake while the Python
    side waits for frames.
    """
    _attendance_state_defaults()

    # The WebRTC context is stored after the component is rendered.  If this
    # fragment fires before that first render has completed, simply wait for
    # the next scheduled run.
    ctx = st.session_state.get("attendance_webrtc_ctx")
    if ctx is None:
        camera_status_placeholder.markdown("**Camera status:** Starting")
        render_message("info", "Waiting for the camera to connect...", detection_placeholder)
        return

    processor = ctx.video_processor
    now = time.monotonic()

    # Camera not (yet, or no longer) delivering frames.
    if processor is None or not processor.has_recent_frame(FRAME_STALE_SECONDS):
        stale_since = st.session_state.get("attendance_stale_since")
        if stale_since is None:
            stale_since = now
            st.session_state["attendance_stale_since"] = stale_since

        if not st.session_state.get("attendance_ever_connected", False):
            camera_status_placeholder.markdown("**Camera status:** Starting")
            if now - stale_since > CAMERA_READY_TIMEOUT_SECONDS:
                render_message("warning", MSG_CAMERA_TIMEOUT, detection_placeholder)
            else:
                render_message("info", "Waiting for the camera to connect...", detection_placeholder)
        else:
            camera_status_placeholder.markdown("**Camera status:** Reconnecting")
            render_message("warning", MSG_CAMERA_LOST, detection_placeholder)

        st.session_state["attendance_countdown_deadline"] = None
        return

    st.session_state["attendance_ever_connected"] = True
    st.session_state["attendance_stale_since"] = None
    camera_status_placeholder.markdown("**Camera status:** Active")

    # Briefly keep the last result visible before scanning again.
    if now < float(st.session_state.get("attendance_hold_until", 0.0)):
        return

    if not load_employees():
        st.session_state["attendance_countdown_deadline"] = None
        render_message("info", MSG_NO_EMPLOYEES, detection_placeholder)
        return

    frame = processor.get_latest_frame_bgr()
    face_count = count_valid_faces(frame)

    if face_count == 0:
        st.session_state["attendance_countdown_deadline"] = None
        render_message("info", MSG_NO_FACE, detection_placeholder)
        return

    if face_count > 1:
        st.session_state["attendance_countdown_deadline"] = None
        render_message("warning", MSG_MULTIPLE_FACES, detection_placeholder)
        return

    # Exactly one valid face: start/continue the existing 5-second countdown.
    deadline = st.session_state.get("attendance_countdown_deadline")
    if deadline is None:
        deadline = now + COUNTDOWN_SECONDS
        st.session_state["attendance_countdown_deadline"] = deadline

    remaining = float(deadline) - now
    if remaining > 0.05:
        render_countdown(detection_placeholder, max(1, int(remaining) + 1))
        return

    render_message("info", "Verifying. Please wait.", detection_placeholder)
    capture_frame = processor.get_latest_frame_bgr()
    if capture_frame is None:
        st.session_state["attendance_countdown_deadline"] = None
        return

    try:
        outcome = process_captured_frame(capture_frame)
    except Exception:
        logger.exception("Unexpected error while processing the captured frame")
        outcome = make_outcome("error", MSG_VERIFICATION_ERROR, captured=True)

    render_outcome(detection_placeholder, outcome)
    render_dashboard(dashboard_placeholder)

    # Reset and keep watching. The WebRTC camera is not stopped.
    st.session_state["attendance_countdown_deadline"] = None
    st.session_state["attendance_hold_until"] = time.monotonic() + RESULT_DISPLAY_SECONDS


def render_attendance_page(dashboard_placeholder: Any) -> None:
    st.subheader("Employee Attendance")

    _attendance_state_defaults()

    _, center, _ = st.columns([1, 2, 1])
    with center:
        try:
            # IMPORTANT: do not run a while-loop after this call.  The browser
            # side of streamlit-webrtc needs the main Streamlit script to return
            # so its offer/answer and camera-permission handshake can complete.
            # desired_playing_state=True keeps the final app fully automatic;
            # the user never needs to press START.
            ctx = webrtc_streamer(
                key=WEBRTC_KEY,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=get_rtc_configuration(),
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                desired_playing_state=True,
                media_toggle_controls=False,
                video_processor_factory=LatestFrameProcessor,
                async_processing=True,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": False,
                    "muted": True,
                    "playsInline": True,
                    "style": {"width": "100%"},
                },
            )
        except Exception:
            logger.exception("WebRTC initialisation failed")
            render_message(
                "error",
                "The camera could not be started. Please check that camera access is allowed "
                "in your browser, then refresh this page.",
            )
            return

        st.session_state["attendance_webrtc_ctx"] = ctx

        camera_status_placeholder = st.empty()
        detection_placeholder = st.empty()
        st.caption(
            "Automatically capturing when one valid face is detected. "
            "Countdown appears only when exactly one face is detected."
        )

        # Non-blocking monitor. The fragment reruns every 0.35 s while the
        # WebRTC component remains free to connect and stream frames.
        _attendance_monitor_fragment(
            dashboard_placeholder,
            camera_status_placeholder,
            detection_placeholder,
        )


# =============================================================================
# EMPLOYEE REGISTRATION PAGE
# =============================================================================
def _new_registration_state() -> Dict[str, Any]:
    return {
        "step": 1,
        "employee_id": "",
        "employee_name": "",
        "images": {view: None for view in FACE_VIEWS},        # JPEG/PNG bytes
        "embeddings": {view: None for view in FACE_VIEWS},    # list of floats
        "rejected": {},                                        # view -> {hash, message}
        "camera_nonce": {view: 0 for view in FACE_VIEWS},      # bump to reset a camera
    }


def init_registration_state() -> None:
    if REG_STATE_KEY not in st.session_state:
        st.session_state[REG_STATE_KEY] = _new_registration_state()
    st.session_state.setdefault(REG_EPOCH_KEY, 0)


def _reset_registration() -> None:
    st.session_state[REG_STATE_KEY] = _new_registration_state()
    st.session_state[REG_EPOCH_KEY] = st.session_state.get(REG_EPOCH_KEY, 0) + 1


def _go_to_step(step: int) -> None:
    st.session_state[REG_STATE_KEY]["step"] = step


def _retake_photo(view: str) -> None:
    reg = st.session_state[REG_STATE_KEY]
    reg["images"][view] = None
    reg["embeddings"][view] = None
    reg["rejected"].pop(view, None)
    reg["camera_nonce"][view] += 1


def registration_photo_message(code: str) -> str:
    return {
        "no_face": "No face was detected in this photo. Please retake it and look at the camera.",
        "multiple_faces": (
            "More than one face was detected. Please retake the photo with only the employee visible."
        ),
        "invalid_image": "This photo could not be read. Please retake it.",
    }.get(code, "This photo could not be processed. Please retake it.")


def validate_and_store_photo(reg: Dict[str, Any], view: str, data: bytes) -> Optional[str]:
    """
    Check that the photo holds exactly one face and keep it with its embedding.
    Returns an error message, or None when the photo was accepted.
    """
    digest = hashlib.sha1(data).hexdigest()
    rejected = reg["rejected"].get(view)
    if rejected and rejected["hash"] == digest:
        return rejected["message"]  # same photo as before; do not run DeepFace again

    image = decode_image(data)
    if image is None:
        error = registration_photo_message("invalid_image")
    else:
        try:
            embedding = generate_embedding(image)
        except FaceProcessingError as exc:
            error = registration_photo_message(exc.code)
        else:
            reg["images"][view] = data
            reg["embeddings"][view] = embedding
            reg["rejected"].pop(view, None)
            return None

    reg["rejected"][view] = {"hash": digest, "message": error}
    return error


def validate_employee_fields(employee_id: str, employee_name: str) -> Optional[str]:
    if not employee_id:
        return "Please enter the Employee ID."
    if not EMPLOYEE_ID_PATTERN.match(employee_id):
        return (
            "The Employee ID may contain only letters, numbers, hyphens and underscores "
            "(maximum 32 characters)."
        )
    if not employee_name:
        return "Please enter the employee name."
    return None


def create_employee_profile(reg: Dict[str, Any]) -> Optional[str]:
    """Generate the embeddings and save the profile. Returns an error message or None."""
    images: Dict[str, np.ndarray] = {}
    embeddings: Dict[str, List[float]] = {}
    for view in FACE_VIEWS:
        image = decode_image(reg["images"][view])
        if image is None:
            return f"The {VIEW_LABELS[view].lower()} photo is missing or unreadable. Please retake it."
        embedding = reg["embeddings"][view]
        if embedding is None:
            try:
                embedding = generate_embedding(image)
            except FaceProcessingError as exc:
                return f"{VIEW_LABELS[view]}: {registration_photo_message(exc.code)}"
        images[view] = image
        embeddings[view] = embedding

    try:
        register_employee(reg["employee_id"], reg["employee_name"], images, embeddings)
    except OSError:
        logger.exception("The employee profile could not be saved")
        return (
            "The employee profile could not be saved. Please check that the application "
            "can write to its folder."
        )
    return None


def render_capture_step(reg: Dict[str, Any], step: int) -> None:
    view = STEP_VIEWS[step]
    epoch = st.session_state[REG_EPOCH_KEY]

    st.markdown(f"**Step {step} of 3: {VIEW_LABELS[view]}**")
    st.caption(VIEW_INSTRUCTIONS[view])

    employee_id_input = employee_name_input = ""
    if step == 1:
        col_id, col_name = st.columns(2)
        employee_id_input = col_id.text_input(
            "Employee ID",
            value=reg["employee_id"],
            key=f"reg_employee_id_{epoch}",
            max_chars=32,
            placeholder="For example EMP001",
        )
        employee_name_input = col_name.text_input(
            "Employee Name",
            value=reg["employee_name"],
            key=f"reg_employee_name_{epoch}",
            max_chars=80,
        )

    stored_photo = reg["images"][view]
    camera_photo = None
    _, center, _ = st.columns([1, 2, 1])
    with center:
        if stored_photo is not None:
            st.image(stored_photo, caption=f"{VIEW_LABELS[view]} photo accepted", use_container_width=True)
        else:
            camera_photo = st.camera_input(
                f"{VIEW_LABELS[view]} camera",
                key=f"reg_camera_{view}_{epoch}_{reg['camera_nonce'][view]}",
                label_visibility="collapsed",
            )
            if camera_photo is not None:
                with st.spinner("Checking the photo..."):
                    error = validate_and_store_photo(reg, view, camera_photo.getvalue())
                if error is None:
                    st.rerun()
                render_message("error", error)

    col_back, col_retake, col_next = st.columns(3)
    col_back.button(
        "Back",
        key=f"reg_back_{step}_{epoch}",
        disabled=step == 1,
        on_click=_go_to_step,
        args=(step - 1,),
        use_container_width=True,
    )
    col_retake.button(
        "Retake This Photo",
        key=f"reg_retake_{step}_{epoch}",
        disabled=stored_photo is None and camera_photo is None,
        on_click=_retake_photo,
        args=(view,),
        use_container_width=True,
    )
    next_clicked = col_next.button(
        "Continue to Review" if step == 3 else "Continue to Next Step",
        key=f"reg_next_{step}_{epoch}",
        type="primary",
        disabled=reg["images"][view] is None,
        use_container_width=True,
    )

    if next_clicked:
        if step == 1:
            employee_id = employee_id_input.strip().upper()
            employee_name = " ".join(employee_name_input.split())
            problem = validate_employee_fields(employee_id, employee_name)
            if problem:
                render_message("error", problem)
                return
            reg["employee_id"] = employee_id
            reg["employee_name"] = employee_name
        reg["step"] = step + 1
        st.rerun()


def render_review_step(reg: Dict[str, Any]) -> None:
    epoch = st.session_state[REG_EPOCH_KEY]
    st.markdown("**Registration Review**")
    st.write(f"Employee ID: **{reg['employee_id']}**")
    st.write(f"Employee Name: **{reg['employee_name']}**")

    for col, view in zip(st.columns(3), FACE_VIEWS):
        with col:
            st.image(reg["images"][view], caption=f"{VIEW_LABELS[view]} Photo", use_container_width=True)

    already_registered = reg["employee_id"] in load_employees()
    replace_confirmed = True
    if already_registered:
        render_message(
            "warning",
            "This Employee ID is already registered. Creating the profile will replace it.",
        )
        replace_confirmed = st.checkbox(
            "Replace the existing profile for this Employee ID", key=f"reg_replace_{epoch}"
        )

    col_back, col_restart, col_create = st.columns(3)
    col_back.button(
        "Back",
        key=f"reg_back_review_{epoch}",
        on_click=_go_to_step,
        args=(3,),
        use_container_width=True,
    )
    col_restart.button(
        "Start Over",
        key=f"reg_restart_{epoch}",
        on_click=_reset_registration,
        use_container_width=True,
    )
    create_clicked = col_create.button(
        "Create Employee Face Profile",
        key=f"reg_create_{epoch}",
        type="primary",
        disabled=not replace_confirmed,
        use_container_width=True,
    )

    if create_clicked:
        with st.spinner("Creating the face profile..."):
            error = create_employee_profile(reg)
        if error is None:
            st.session_state[REG_NOTICE_KEY] = (
                "success",
                f"Face profile created for {reg['employee_name']} ({reg['employee_id']}).",
            )
            _reset_registration()
            st.rerun()
        render_message("error", error)


def render_registered_employees() -> None:
    employees = load_employees()
    with st.expander("Registered Employees", expanded=False):
        if not employees:
            st.write("No employees have been registered yet.")
            return
        rows = [
            {
                "Employee ID": employee_id,
                "Employee Name": record.get("name", ""),
                "Registered At": record.get("registered_at", ""),
            }
            for employee_id, record in sorted(employees.items())
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_registration_page() -> None:
    init_registration_state()
    reg = st.session_state[REG_STATE_KEY]

    st.subheader("Employee Registration")
    notice = st.session_state.pop(REG_NOTICE_KEY, None)
    if notice:
        render_message(notice[0], notice[1])

    step = reg["step"]
    if step in STEP_VIEWS:
        st.progress(step / REVIEW_STEP, text=f"Step {step} of 3")
        render_capture_step(reg, step)
    else:
        st.progress(1.0, text="Review")
        render_review_step(reg)

    render_registered_employees()


# =============================================================================
# ATTENDANCE RECORDS PAGE
# =============================================================================
ALL_EMPLOYEES = "All Employees"
ALL_DATES = "All Dates"


def render_records_page() -> None:
    st.subheader("Attendance Records")
    attendance = load_attendance()
    employees = load_employees()
    today = now_local().strftime(DATE_FORMAT)

    # Employee filter options: "Name (ID)" -> ID
    names: Dict[str, str] = {}
    for employee_id, record in employees.items():
        names[employee_id] = str(record.get("name") or employee_id)
    if not attendance.empty:
        for employee_id, name in attendance[["Employee ID", "Employee Name"]].drop_duplicates().itertuples(index=False):
            names.setdefault(str(employee_id), str(name))
    employee_options = {ALL_EMPLOYEES: ""}
    for employee_id in sorted(names):
        employee_options[f"{names[employee_id]} ({employee_id})"] = employee_id

    dates = set(attendance["Date"].unique()) if not attendance.empty else set()
    dates.add(today)
    date_options = [ALL_DATES] + sorted(dates, reverse=True)

    col_employee, col_date = st.columns(2)
    employee_label = col_employee.selectbox(
        "Employee", list(employee_options), key="records_employee_filter"
    )
    date_label = col_date.selectbox(
        "Date", date_options, index=date_options.index(today), key="records_date_filter"
    )
    employee_id_filter = employee_options.get(employee_label, "")

    filtered = attendance
    if employee_id_filter:
        filtered = filtered[filtered["Employee ID"] == employee_id_filter]
    if date_label != ALL_DATES:
        filtered = filtered[filtered["Date"] == date_label]
    filtered = filtered.sort_values(["Date", "Time"], ascending=False, kind="stable")

    if filtered.empty:
        render_message("info", "No attendance records found for the selected filters.")
    else:
        st.dataframe(filtered, hide_index=True, use_container_width=True)

    file_date = "all_dates" if date_label == ALL_DATES else date_label
    st.download_button(
        "Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"attendance_{file_date}.csv",
        mime="text/csv",
        key="records_download_csv",
    )

    summary_date = today if date_label == ALL_DATES else date_label
    st.markdown(f"**Daily Summary ({summary_date})**")
    summary = calculate_daily_summary(attendance, summary_date, employee_id_filter or None)
    if summary.empty:
        st.write("No attendance events for this date.")
    else:
        st.dataframe(summary, hide_index=True, use_container_width=True)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    inject_styles()

    st.markdown(
        f'<div class="app-header"><div class="app-title">{html.escape(APP_TITLE)}</div>'
        '<div class="app-subtitle">Contactless employee attendance</div></div>',
        unsafe_allow_html=True,
    )

    if not ensure_storage():
        render_message(
            "error",
            "The data files could not be created. Please check that the application can "
            "write to its folder.",
        )
        st.stop()

    dashboard_placeholder = st.empty()
    render_dashboard(dashboard_placeholder)

    warm_up_face_engine()

    section = st.radio(
        "Section",
        NAV_SECTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )

    # Only one camera may be active at a time, so the live attendance camera is
    # released whenever another section is shown.
    if section != NAV_ATTENDANCE:
        reset_attendance_session()

    if section == NAV_ATTENDANCE:
        render_attendance_page(dashboard_placeholder)
    elif section == NAV_REGISTRATION:
        render_registration_page()
    else:
        render_records_page()


if __name__ == "__main__":
    main()