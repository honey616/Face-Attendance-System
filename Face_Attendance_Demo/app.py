import os

# TensorFlow / native library stability settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import json
import tempfile
import threading
import time
from datetime import datetime, date

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)
from streamlit_autorefresh import st_autorefresh


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Employee Attendance System"
APP_VERSION = "3.1.0"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"

MATCH_THRESHOLD = 0.30

# Same employee cannot create another event within this time
SCAN_COOLDOWN_SECONDS = 60

# Process one frame every N frames
PROCESS_EVERY_N_FRAMES = 20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
EMPLOYEES_FILE = os.path.join(BASE_DIR, "employees.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .app-header {
        padding: 20px 24px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 20px;
    }

    .app-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .status-box {
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .status-title {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 5px;
    }

    .status-value {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
    }

    .employee-card {
        padding: 18px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 15px;
    }

    .employee-name {
        font-size: 21px;
        font-weight: 700;
        color: #111827;
    }

    .employee-id {
        color: #6b7280;
        font-size: 14px;
        margin-top: 3px;
    }

    .event-checkin {
        color: #166534;
        font-weight: 700;
    }

    .event-checkout {
        color: #991b1b;
        font-weight: 700;
    }

    .info-box {
        padding: 14px 16px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        color: #374151;
        font-size: 14px;
        line-height: 1.6;
    }

    .footer-text {
        color: #9ca3af;
        font-size: 12px;
        text-align: center;
        padding-top: 30px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
        background: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-title">{APP_NAME}</div>
        <div class="app-subtitle">
            Automated face-based employee check-in and check-out
            &nbsp; | &nbsp; Version {APP_VERSION}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA FILE INITIALIZATION
# ============================================================

def initialize_files():
    """Create required JSON and CSV files if they don't exist."""

    if not os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)

    if not os.path.exists(ATTENDANCE_FILE):
        columns = [
            "employee_id",
            "employee_name",
            "event",
            "timestamp",
            "date",
        ]

        df = pd.DataFrame(columns=columns)
        df.to_csv(ATTENDANCE_FILE, index=False)


initialize_files()


# ============================================================
# EMPLOYEE STORAGE
# ============================================================

def load_employees():
    try:
        with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception:
        return {}


def save_employees(data):
    with open(EMPLOYEES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_attendance():
    try:
        df = pd.read_csv(ATTENDANCE_FILE)

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "employee_id",
                    "employee_name",
                    "event",
                    "timestamp",
                    "date",
                ]
            )

        return df

    except Exception:
        return pd.DataFrame(
            columns=[
                "employee_id",
                "employee_name",
                "event",
                "timestamp",
                "date",
            ]
        )


def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)


# ============================================================
# DEEPFACE - LAZY IMPORT
# ============================================================

@st.cache_resource(show_spinner=False)
def load_deepface():
    """
    DeepFace is intentionally imported only when required.
    This prevents Streamlit Cloud from loading TensorFlow/
    DeepFace during initial application startup.
    """

    from deepface import DeepFace

    return DeepFace


# ============================================================
# FACE EMBEDDING
# ============================================================

def generate_embedding(image_path):

    DeepFace = load_deepface()

    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if not result:
        raise ValueError("No face detected.")

    if len(result) != 1:
        raise ValueError(
            "Please keep only one face in front of the camera."
        )

    embedding = np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )

    return embedding


# ============================================================
# IMAGE HELPERS
# ============================================================

def save_uploaded_image(uploaded_file):

    suffix = ".jpg"

    if hasattr(uploaded_file, "name"):
        extension = os.path.splitext(uploaded_file.name)[1]

        if extension:
            suffix = extension

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


def save_numpy_image(image):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        cv2.imwrite(temp_file.name, image)

        return temp_file.name


# ============================================================
# COSINE DISTANCE
# ============================================================

def cosine_distance(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 1.0

    similarity = np.dot(a, b) / denominator

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    return float(1.0 - similarity)


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_employee(embedding):

    employees = load_employees()

    if not employees:
        return None, None

    best_employee = None
    best_distance = float("inf")

    for employee_id, employee_data in employees.items():

        stored_embedding = employee_data.get("embedding")

        if not stored_embedding:
            continue

        try:
            distance = cosine_distance(
                embedding,
                stored_embedding
            )

        except Exception:
            continue

        if distance < best_distance:

            best_distance = distance
            best_employee = employee_id

    if best_employee is None:
        return None, None

    if best_distance <= MATCH_THRESHOLD:

        employee = employees[best_employee]

        return {
            "employee_id": best_employee,
            "employee_name": employee.get(
                "name",
                best_employee
            ),
        }, best_distance

    return None, best_distance


# ============================================================
# EVENT LOGIC
# ============================================================

def get_employee_events(employee_id):

    df = load_attendance()

    if df.empty:
        return df

    return df[
        df["employee_id"].astype(str)
        == str(employee_id)
    ].copy()


def get_next_event(employee_id):

    employee_events = get_employee_events(
        employee_id
    )

    if employee_events.empty:
        return "CHECK-IN"

    employee_events = employee_events.sort_values(
        "timestamp"
    )

    last_event = str(
        employee_events.iloc[-1]["event"]
    ).upper()

    if last_event == "CHECK-IN":
        return "CHECK-OUT"

    return "CHECK-IN"


def check_cooldown(employee_id):

    df = get_employee_events(employee_id)

    if df.empty:
        return True, 0

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp_dt"]
    )

    if df.empty:
        return True, 0

    last_timestamp = df.iloc[-1]["timestamp_dt"]

    now = datetime.now()

    elapsed = (
        now - last_timestamp.to_pydatetime()
    ).total_seconds()

    if elapsed < SCAN_COOLDOWN_SECONDS:

        remaining = int(
            SCAN_COOLDOWN_SECONDS - elapsed
        )

        return False, remaining

    return True, 0


def record_event(employee_id, employee_name):

    allowed, remaining = check_cooldown(
        employee_id
    )

    if not allowed:

        return {
            "success": False,
            "message": (
                f"Please wait {remaining} seconds "
                f"before the next scan."
            ),
            "event": None,
        }

    next_event = get_next_event(employee_id)

    now = datetime.now()

    new_row = {
        "employee_id": employee_id,
        "employee_name": employee_name,
        "event": next_event,
        "timestamp": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "date": now.strftime("%Y-%m-%d"),
    }

    df = load_attendance()

    df = pd.concat(
        [
            df,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

    save_attendance(df)

    return {
        "success": True,
        "message": (
            f"{employee_name} "
            f"{next_event.replace('-', ' ')} "
            f"successfully."
        ),
        "event": next_event,
    }


# ============================================================
# DAILY WORKING HOURS
# ============================================================

def calculate_employee_working_time(
    employee_id,
    target_date=None
):

    if target_date is None:
        target_date = date.today()

    df = load_attendance()

    if df.empty:
        return 0, 0

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df[
        df["employee_id"].astype(str)
        == str(employee_id)
    ]

    df = df[
        df["timestamp_dt"].dt.date
        == target_date
    ]

    if df.empty:
        return 0, 0

    df = df.sort_values(
        "timestamp_dt"
    )

    total_seconds = 0
    visits = 0

    check_in_time = None

    for _, row in df.iterrows():

        event = str(
            row["event"]
        ).upper()

        timestamp = row["timestamp_dt"]

        if event == "CHECK-IN":

            check_in_time = timestamp

        elif (
            event == "CHECK-OUT"
            and check_in_time is not None
        ):

            seconds = (
                timestamp - check_in_time
            ).total_seconds()

            if seconds >= 0:

                total_seconds += seconds
                visits += 1

            check_in_time = None

    return total_seconds, visits


def format_duration(seconds):

    seconds = int(seconds)

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    return f"{hours}h {minutes}m"


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class FaceCameraProcessor(VideoProcessorBase):

    def __init__(self):

        self.frame_count = 0

        self.latest_frame = None

        self.lock = threading.Lock()

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        if (
            self.frame_count
            % PROCESS_EVERY_N_FRAMES
            == 0
        ):

            with self.lock:

                self.latest_frame = (
                    image.copy()
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "last_recognition" not in st.session_state:
    st.session_state.last_recognition = None

if "last_distance" not in st.session_state:
    st.session_state.last_distance = None

if "last_event_message" not in st.session_state:
    st.session_state.last_event_message = None

if "last_event_type" not in st.session_state:
    st.session_state.last_event_type = None

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### Employee Attendance"
    )

    page = st.radio(
        "Navigation",
        [
            "Live Attendance",
            "Register Employee",
            "Attendance History",
            "Employees",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        **System**

        Face recognition: Active  
        Attendance mode: Automatic  
        Event sequence: Check-in / Check-out  
        Duplicate protection: 60 seconds
        """
    )


# ============================================================
# LIVE ATTENDANCE
# ============================================================

if page == "Live Attendance":

    st.markdown(
        '<div class="section-title">'
        'Live Attendance'
        '</div>',
        unsafe_allow_html=True
    )

    employees = load_employees()

    if not employees:

        st.warning(
            "No employees registered. "
            "Please register an employee first."
        )

    else:

        col1, col2 = st.columns(
            [2.2, 1]
        )

        with col1:

            st.markdown(
                """
                <div class="info-box">
                    Stand in front of the camera.
                    The system will automatically detect
                    and recognize the employee.
                    No manual attendance button is required.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            webrtc_ctx = webrtc_streamer(
                key="employee-attendance-camera",

                mode=WebRtcMode.SENDRECV,

                video_processor_factory=(
                    FaceCameraProcessor
                ),

                media_stream_constraints={
                    "video": {
                        "width": 640,
                        "height": 480
                    },
                    "audio": False,
                },

                async_processing=True,
            )

            if webrtc_ctx.state.playing:

                st.session_state.camera_running = True

                # Refresh UI every second
                st_autorefresh(
                    interval=1000,
                    key="attendance_refresh"
                )

                processor = (
                    webrtc_ctx.video_processor
                )

                if processor is not None:

                    frame = None

                    with processor.lock:

                        if (
                            processor.latest_frame
                            is not None
                        ):

                            frame = (
                                processor.latest_frame.copy()
                            )

                    if frame is not None:

                        try:

                            # Resize for faster processing
                            height, width = (
                                frame.shape[:2]
                            )

                            max_width = 640

                            if width > max_width:

                                scale = (
                                    max_width
                                    / width
                                )

                                frame = cv2.resize(
                                    frame,
                                    (
                                        int(
                                            width
                                            * scale
                                        ),
                                        int(
                                            height
                                            * scale
                                        ),
                                    ),
                                )

                            image_path = (
                                save_numpy_image(
                                    frame
                                )
                            )

                            try:

                                embedding = (
                                    generate_embedding(
                                        image_path
                                    )
                                )

                            finally:

                                try:
                                    os.remove(
                                        image_path
                                    )
                                except Exception:
                                    pass

                            employee, distance = (
                                recognize_employee(
                                    embedding
                                )
                            )

                            if employee:

                                employee_id = (
                                    employee[
                                        "employee_id"
                                    ]
                                )

                                employee_name = (
                                    employee[
                                        "employee_name"
                                    ]
                                )

                                st.session_state.last_recognition = (
                                    employee
                                )

                                st.session_state.last_distance = (
                                    distance
                                )

                                result = record_event(
                                    employee_id,
                                    employee_name
                                )

                                if result["success"]:

                                    st.session_state.last_event_message = (
                                        result[
                                            "message"
                                        ]
                                    )

                                    st.session_state.last_event_type = (
                                        result[
                                            "event"
                                        ]
                                    )

                                else:

                                    st.session_state.last_event_message = (
                                        result[
                                            "message"
                                        ]
                                    )

                                    st.session_state.last_event_type = (
                                        None
                                    )

                        except Exception as e:

                            error_text = str(e)

                            if (
                                "No face detected"
                                not in error_text
                            ):

                                st.session_state.last_event_message = (
                                    "Face processing error: "
                                    + error_text
                                )

        with col2:

            st.markdown(
                '<div class="section-title">'
                'Current Status'
                '</div>',
                unsafe_allow_html=True
            )

            if (
                st.session_state.last_recognition
                is not None
            ):

                employee = (
                    st.session_state.last_recognition
                )

                st.markdown(
                    f"""
                    <div class="employee-card">
                        <div class="employee-name">
                            {employee["employee_name"]}
                        </div>
                        <div class="employee-id">
                            Employee ID:
                            {employee["employee_id"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                distance = (
                    st.session_state.last_distance
                )

                if distance is not None:

                    st.metric(
                        "Face Distance",
                        f"{distance:.4f}"
                    )

                message = (
                    st.session_state.last_event_message
                )

                if message:

                    event_type = (
                        st.session_state.last_event_type
                    )

                    if event_type == "CHECK-IN":

                        st.success(message)

                    elif event_type == "CHECK-OUT":

                        st.info(message)

                    else:

                        st.warning(message)

                employee_id = (
                    employee[
                        "employee_id"
                    ]
                )

                total_seconds, visits = (
                    calculate_employee_working_time(
                        employee_id
                    )
                )

                st.metric(
                    "Today's Working Time",
                    format_duration(
                        total_seconds
                    )
                )

                st.metric(
                    "Completed Visits",
                    visits
                )

                next_event = (
                    get_next_event(
                        employee_id
                    )
                )

                st.markdown(
                    f"""
                    <div class="status-box">
                        <div class="status-title">
                            Next attendance event
                        </div>
                        <div class="status-value">
                            {next_event.replace("-", " ")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="status-box">
                        <div class="status-title">
                            Recognition Status
                        </div>
                        <div class="status-value">
                            Waiting for employee
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# REGISTER EMPLOYEE
# ============================================================

elif page == "Register Employee":

    st.markdown(
        '<div class="section-title">'
        'Register Employee'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            Register one employee at a time.
            Capture a clear front-facing image with
            only one person visible.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        employee_id = st.text_input(
            "Employee ID",
            placeholder="EMP001"
        )

        employee_name = st.text_input(
            "Employee Name",
            placeholder="Employee full name"
        )

    with col2:

        camera_image = st.camera_input(
            "Capture Employee Face"
        )

    if st.button(
        "Register Employee",
        type="primary",
        use_container_width=True
    ):

        if not employee_id.strip():

            st.error(
                "Please enter Employee ID."
            )

        elif not employee_name.strip():

            st.error(
                "Please enter Employee Name."
            )

        elif camera_image is None:

            st.error(
                "Please capture employee image."
            )

        else:

            employees = load_employees()

            employee_id = (
                employee_id.strip()
            )

            employee_name = (
                employee_name.strip()
            )

            if employee_id in employees:

                st.error(
                    "Employee ID already exists."
                )

            else:

                image_path = None

                try:

                    image_path = (
                        save_uploaded_image(
                            camera_image
                        )
                    )

                    embedding = (
                        generate_embedding(
                            image_path
                        )
                    )

                    face_image_path = os.path.join(
                        KNOWN_FACES_DIR,
                        f"{employee_id}.jpg"
                    )

                    image_bytes = (
                        camera_image.getvalue()
                    )

                    with open(
                        face_image_path,
                        "wb"
                    ) as f:

                        f.write(
                            image_bytes
                        )

                    employees[
                        employee_id
                    ] = {

                        "name":
                            employee_name,

                        "embedding":
                            embedding.tolist(),

                        "registered_at":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "image":
                            face_image_path,
                    }

                    save_employees(
                        employees
                    )

                    st.success(
                        f"{employee_name} "
                        "registered successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Registration failed: {e}"
                    )

                finally:

                    if image_path:

                        try:
                            os.remove(
                                image_path
                            )
                        except Exception:
                            pass


# ============================================================
# ATTENDANCE HISTORY
# ============================================================

elif page == "Attendance History":

    st.markdown(
        '<div class="section-title">'
        'Attendance History'
        '</div>',
        unsafe_allow_html=True
    )

    df = load_attendance()

    if df.empty:

        st.info(
            "No attendance records available."
        )

    else:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.sort_values(
            "timestamp",
            ascending=False
        )

        col1, col2, col3 = st.columns(
            [1, 1, 1]
        )

        with col1:

            selected_date = st.date_input(
                "Date",
                value=date.today()
            )

        with col2:

            employee_options = [
                "All Employees"
            ] + sorted(
                df[
                    "employee_name"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_employee = (
                st.selectbox(
                    "Employee",
                    employee_options
                )
            )

        with col3:

            event_options = [
                "All Events",
                "CHECK-IN",
                "CHECK-OUT"
            ]

            selected_event = (
                st.selectbox(
                    "Event",
                    event_options
                )
            )

        filtered = df[
            df["timestamp"].dt.date
            == selected_date
        ].copy()

        if selected_employee != "All Employees":

            filtered = filtered[
                filtered[
                    "employee_name"
                ].astype(str)
                == selected_employee
            ]

        if selected_event != "All Events":

            filtered = filtered[
                filtered["event"]
                == selected_event
            ]

        if filtered.empty:

            st.info(
                "No records found for selected filters."
            )

        else:

            display_df = filtered[
                [
                    "employee_id",
                    "employee_name",
                    "event",
                    "timestamp",
                ]
            ].copy()

            display_df[
                "timestamp"
            ] = display_df[
                "timestamp"
            ].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = (
                filtered.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=(
                    f"attendance_"
                    f"{selected_date}.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# EMPLOYEE MANAGEMENT
# ============================================================

elif page == "Employees":

    st.markdown(
        '<div class="section-title">'
        'Registered Employees'
        '</div>',
        unsafe_allow_html=True
    )

    employees = load_employees()

    if not employees:

        st.info(
            "No employees registered."
        )

    else:

        rows = []

        for employee_id, employee in (
            employees.items()
        ):

            total_seconds, visits = (
                calculate_employee_working_time(
                    employee_id
                )
            )

            rows.append(
                {
                    "Employee ID":
                        employee_id,

                    "Employee Name":
                        employee.get(
                            "name",
                            ""
                        ),

                    "Registered At":
                        employee.get(
                            "registered_at",
                            ""
                        ),

                    "Today's Working Time":
                        format_duration(
                            total_seconds
                        ),

                    "Today's Visits":
                        visits,
                }
            )

        employee_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            employee_df,
            use_container_width=True,
            hide_index=True,
        )

        st.write("")

        st.markdown(
            '<div class="section-title">'
            'Employee Details'
            '</div>',
            unsafe_allow_html=True
        )

        selected_id = st.selectbox(
            "Select Employee",
            list(employees.keys()),
            format_func=lambda x:
                f"{x} - {employees[x].get('name', '')}"
        )

        if selected_id:

            employee = (
                employees[selected_id]
            )

            col1, col2 = st.columns(
                [1, 2]
            )

            with col1:

                image_path = employee.get(
                    "image"
                )

                if (
                    image_path
                    and os.path.exists(
                        image_path
                    )
                ):

                    st.image(
                        image_path,
                        width=220
                    )

            with col2:

                st.markdown(
                    f"""
                    <div class="employee-card">

                        <div class="employee-name">
                            {employee.get("name", "")}
                        </div>

                        <div class="employee-id">
                            Employee ID:
                            {selected_id}
                        </div>

                        <br>

                        <div>
                            Registered:
                            {employee.get("registered_at", "")}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "Delete Employee",
                    type="secondary"
                ):

                    del employees[
                        selected_id
                    ]

                    save_employees(
                        employees
                    )

                    image_path = os.path.join(
                        KNOWN_FACES_DIR,
                        f"{selected_id}.jpg"
                    )

                    if os.path.exists(
                        image_path
                    ):

                        try:
                            os.remove(
                                image_path
                            )
                        except Exception:
                            pass

                    st.success(
                        "Employee deleted successfully."
                    )

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        Employee Attendance System
        | Automated Face Recognition
    </div>
    """,
    unsafe_allow_html=True
)