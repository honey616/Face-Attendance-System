import os
import json
import time
import queue
import threading
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Employee Attendance System"
APP_VERSION = "3.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMPLOYEES_FILE = os.path.join(BASE_DIR, "employees.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"

MATCH_THRESHOLD = 0.30
SCAN_COOLDOWN_SECONDS = 60
PROCESS_INTERVAL_SECONDS = 1.5

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# RESPONSIVE UI
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #667085;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .live-card {
        border: 1px solid #D0D5DD;
        border-radius: 14px;
        padding: 14px;
        background: #FFFFFF;
        margin-bottom: 16px;
    }

    .result-card {
        border: 1px solid #D0D5DD;
        border-radius: 12px;
        padding: 14px;
        margin-top: 12px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            padding-top: 1rem;
        }

        .app-title {
            font-size: 24px;
        }

        .app-subtitle {
            font-size: 13px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOCAL STORAGE
# ============================================================

def initialize_storage():
    if not os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(
            columns=[
                "Employee ID",
                "Employee Name",
                "Date",
                "Event",
                "Time",
            ]
        ).to_csv(ATTENDANCE_FILE, index=False)


initialize_storage()


def load_employees():
    try:
        with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_employees(data):
    with open(EMPLOYEES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_attendance():
    columns = [
        "Employee ID",
        "Employee Name",
        "Date",
        "Event",
        "Time",
    ]

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
    except Exception:
        df = pd.DataFrame(columns=columns)

    if not all(c in df.columns for c in columns):
        df = pd.DataFrame(columns=columns)

    return df


# ============================================================
# FACE RECOGNITION
# ============================================================

def generate_embedding(image):
    result = DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if not result:
        raise ValueError("No face detected.")

    if len(result) != 1:
        raise ValueError("Please keep exactly one face in the camera.")

    return np.asarray(
        result[0]["embedding"],
        dtype=np.float32,
    )


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 1.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    similarity = np.clip(similarity, -1.0, 1.0)

    return float(1.0 - similarity)


def recognize_employee(query_embedding):
    employees = load_employees()

    best_employee = None
    best_distance = float("inf")

    for employee_id, employee in employees.items():
        stored = employee.get("embedding")

        if not stored:
            continue

        try:
            distance = cosine_distance(
                query_embedding,
                np.asarray(stored, dtype=np.float32),
            )

            if distance < best_distance:
                best_distance = distance
                best_employee = employee_id

        except Exception:
            continue

    if (
        best_employee is not None
        and best_distance <= MATCH_THRESHOLD
    ):
        return best_employee, best_distance

    return None, best_distance


# ============================================================
# ATTENDANCE LOGIC
# ============================================================

def get_next_event(employee_id, current_date):
    df = load_attendance()

    rows = df[
        (df["Employee ID"].astype(str) == str(employee_id))
        & (df["Date"].astype(str) == current_date)
    ]

    if rows.empty:
        return "CHECK-IN"

    last_event = str(rows.iloc[-1]["Event"])

    return "CHECK-OUT" if last_event == "CHECK-IN" else "CHECK-IN"


def cooldown_active(employee_id, current_date, now):
    df = load_attendance()

    rows = df[
        (df["Employee ID"].astype(str) == str(employee_id))
        & (df["Date"].astype(str) == current_date)
    ]

    if rows.empty:
        return False, 0

    last_time = str(rows.iloc[-1]["Time"])

    try:
        last_dt = datetime.strptime(
            f"{current_date} {last_time}",
            "%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        return False, 0

    elapsed = (now - last_dt).total_seconds()

    if elapsed < SCAN_COOLDOWN_SECONDS:
        return True, max(
            1,
            int(SCAN_COOLDOWN_SECONDS - elapsed),
        )

    return False, 0


def record_attendance(employee_id):
    employees = load_employees()

    if employee_id not in employees:
        return {
            "ok": False,
            "message": "Employee is not registered.",
        }

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    active, remaining = cooldown_active(
        employee_id,
        current_date,
        now,
    )

    if active:
        return {
            "ok": False,
            "message": (
                f"Please wait {remaining} seconds "
                "before the next scan."
            ),
        }

    event = get_next_event(
        employee_id,
        current_date,
    )

    employee_name = employees[employee_id]["name"]

    df = load_attendance()

    new_row = pd.DataFrame(
        [
            {
                "Employee ID": employee_id,
                "Employee Name": employee_name,
                "Date": current_date,
                "Event": event,
                "Time": current_time,
            }
        ]
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True,
    )

    df.to_csv(
        ATTENDANCE_FILE,
        index=False,
    )

    return {
        "ok": True,
        "employee_id": employee_id,
        "employee_name": employee_name,
        "event": event,
        "date": current_date,
        "time": current_time,
    }


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class LiveAttendanceProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()
        self.last_result = {
            "status": "Waiting for face...",
            "employee_name": "",
            "employee_id": "",
            "event": "",
            "time": "",
            "distance": None,
        }
        self.last_processed = 0.0
        self.running = True

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self.worker.start()

    def _set_result(self, result):
        with self.result_lock:
            self.last_result = result

    def get_result(self):
        with self.result_lock:
            return dict(self.last_result)

    def _worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                embedding = generate_embedding(frame)

                employee_id, distance = recognize_employee(
                    embedding
                )

                if employee_id is None:
                    self._set_result(
                        {
                            "status": "Face not recognized",
                            "employee_name": "",
                            "employee_id": "",
                            "event": "",
                            "time": "",
                            "distance": distance,
                        }
                    )
                    continue

                result = record_attendance(employee_id)

                if result["ok"]:
                    self._set_result(
                        {
                            "status": "Attendance recorded",
                            "employee_name": result[
                                "employee_name"
                            ],
                            "employee_id": result[
                                "employee_id"
                            ],
                            "event": result["event"],
                            "time": result["time"],
                            "distance": distance,
                        }
                    )
                else:
                    self._set_result(
                        {
                            "status": result["message"],
                            "employee_name": load_employees()[
                                employee_id
                            ]["name"],
                            "employee_id": employee_id,
                            "event": "",
                            "time": "",
                            "distance": distance,
                        }
                    )

            except ValueError as exc:
                self._set_result(
                    {
                        "status": str(exc),
                        "employee_name": "",
                        "employee_id": "",
                        "event": "",
                        "time": "",
                        "distance": None,
                    }
                )

            except Exception as exc:
                self._set_result(
                    {
                        "status": "Face processing error",
                        "employee_name": "",
                        "employee_id": "",
                        "event": "",
                        "time": "",
                        "distance": None,
                        "error": str(exc),
                    }
                )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        now = time.time()

        if (
            now - self.last_processed
            >= PROCESS_INTERVAL_SECONDS
        ):
            self.last_processed = now

            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()

                self.frame_queue.put_nowait(image.copy())
            except queue.Full:
                pass
            except Exception:
                pass

        return frame

    def stop(self):
        self.running = False


# ============================================================
# DASHBOARD HELPERS
# ============================================================

def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_employee_summary(employee_id, date_string):
    df = load_attendance()

    rows = df[
        (df["Employee ID"].astype(str) == str(employee_id))
        & (df["Date"].astype(str) == date_string)
    ].copy()

    if rows.empty:
        return {
            "first_checkin": "-",
            "last_checkout": "-",
            "visits": 0,
            "working_seconds": 0,
            "status": "OUT",
        }

    rows["dt"] = pd.to_datetime(
        rows["Date"].astype(str)
        + " "
        + rows["Time"].astype(str),
        errors="coerce",
    )

    rows = rows.sort_values("dt")

    first_checkin = "-"
    last_checkout = "-"
    working_seconds = 0
    open_checkin = None
    visits = 0

    for _, row in rows.iterrows():
        event = str(row["Event"])
        dt = row["dt"]

        if pd.isna(dt):
            continue

        if event == "CHECK-IN":
            if first_checkin == "-":
                first_checkin = dt.strftime("%H:%M:%S")
            open_checkin = dt
            visits += 1

        elif event == "CHECK-OUT":
            last_checkout = dt.strftime("%H:%M:%S")

            if open_checkin is not None:
                working_seconds += int(
                    (dt - open_checkin).total_seconds()
                )
                open_checkin = None

    status = "IN" if open_checkin is not None else "OUT"

    if open_checkin is not None:
        working_seconds += int(
            (datetime.now() - open_checkin).total_seconds()
        )

    return {
        "first_checkin": first_checkin,
        "last_checkout": last_checkout,
        "visits": visits,
        "working_seconds": working_seconds,
        "status": status,
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">Employee Attendance System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Live face verification and employee attendance"
    "</div>",
    unsafe_allow_html=True,
)


employees = load_employees()
attendance = load_attendance()

today = datetime.now().strftime("%Y-%m-%d")
today_records = attendance[
    attendance["Date"].astype(str) == today
]


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Registered Employees",
        len(employees),
    )

with col2:
    st.metric(
        "Today's Events",
        len(today_records),
    )

with col3:
    st.metric(
        "System",
        "Online",
    )


st.divider()


# ============================================================
# TABS
# ============================================================

attendance_tab, registration_tab, records_tab = st.tabs(
    [
        "Attendance",
        "Employee Registration",
        "Attendance Records",
    ]
)


# ============================================================
# ATTENDANCE TAB
# ============================================================

with attendance_tab:

    st.subheader("Live Attendance")

    if not employees:
        st.warning(
            "No employees are registered. "
            "Register an employee first."
        )
    else:
        st.markdown(
            '<div class="live-card">'
            "Stand in front of the camera. "
            "The system will automatically recognize "
            "the employee and record the next event."
            "</div>",
            unsafe_allow_html=True,
        )

        RTC_CONFIGURATION = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )

        ctx = webrtc_streamer(
            key="employee-attendance-camera",
            video_processor_factory=LiveAttendanceProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "facingMode": "user",
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_processor:
            result = ctx.video_processor.get_result()

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            st.write(
                f"Status: {result.get('status', 'Waiting...')}"
            )

            if result.get("employee_name"):
                st.write(
                    f"Employee: "
                    f"{result['employee_name']}"
                )

            if result.get("event"):
                st.write(
                    f"Event: {result['event']}"
                )

            if result.get("time"):
                st.write(
                    f"Time: {result['time']}"
                )

            if result.get("distance") is not None:
                st.write(
                    "Face distance: "
                    f"{result['distance']:.4f}"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            "Keep one face centered in the camera. "
            "The system scans approximately every "
            f"{PROCESS_INTERVAL_SECONDS:.1f} seconds."
        )


# ============================================================
# REGISTRATION TAB
# ============================================================

with registration_tab:

    st.subheader("Employee Registration")

    st.write(
        "Use one clear photo containing exactly one employee face."
    )

    employee_id = st.text_input(
        "Employee ID",
        placeholder="EMP001",
    )

    employee_name = st.text_input(
        "Employee Name",
        placeholder="Employee name",
    )

    registration_image = st.camera_input(
        "Capture registration photo"
    )

    if st.button(
        "Register Employee",
        use_container_width=True,
    ):

        if not employee_id.strip():
            st.error("Enter Employee ID.")
        elif not employee_name.strip():
            st.error("Enter Employee Name.")
        elif registration_image is None:
            st.error("Capture a registration photo.")
        else:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp:
                    temp.write(
                        registration_image.getvalue()
                    )
                    temp_path = temp.name

                embedding = generate_embedding(
                    temp_path
                )

                employees = load_employees()

                employees[employee_id.strip()] = {
                    "name": employee_name.strip(),
                    "embedding": embedding.tolist(),
                    "registered_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }

                save_employees(employees)

                st.success(
                    f"{employee_name.strip()} registered successfully."
                )

            except Exception as exc:
                st.error(
                    f"Registration failed: {exc}"
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass


# ============================================================
# RECORDS TAB
# ============================================================

with records_tab:

    st.subheader("Attendance Records")

    df = load_attendance()

    if df.empty:
        st.info("No attendance records available.")
    else:
        st.dataframe(
            df.sort_values(
                ["Date", "Time"],
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Attendance CSV",
            data=csv_data,
            file_name="attendance.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    st.subheader("Today's Employee Summary")

    if employees:
        summary_rows = []

        for employee_id, employee in employees.items():
            summary = calculate_employee_summary(
                employee_id,
                today,
            )

            summary_rows.append(
                {
                    "Employee ID": employee_id,
                    "Employee Name": employee["name"],
                    "First Check-In": summary[
                        "first_checkin"
                    ],
                    "Last Check-Out": summary[
                        "last_checkout"
                    ],
                    "Visits": summary["visits"],
                    "Working Hours": format_duration(
                        summary["working_seconds"]
                    ),
                    "Status": summary["status"],
                }
            )

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
