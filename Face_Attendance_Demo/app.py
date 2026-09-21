import os
import json
import tempfile
import time
from datetime import datetime

import av
import streamlit as st

try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV could not be loaded: {e}")
    st.stop()

import numpy as np
import pandas as pd

from deepface import DeepFace
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
APP_VERSION = "3.0.0"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMPLOYEES_FILE = os.path.join(
    BASE_DIR,
    "employees.json"
)

ATTENDANCE_FILE = os.path.join(
    BASE_DIR,
    "attendance.csv"
)

KNOWN_FACES_DIR = os.path.join(
    BASE_DIR,
    "known_faces"
)

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"

MATCH_THRESHOLD = 0.30

SCAN_COOLDOWN_SECONDS = 60

PROCESS_EVERY_N_FRAMES = 20


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CORPORATE CSS
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
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 650;
        letter-spacing: -0.4px;
        margin-bottom: 3px;
    }

    .app-subtitle {
        color: #667085;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .section-description {
        color: #667085;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .status-online {
        color: #087443;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

os.makedirs(
    KNOWN_FACES_DIR,
    exist_ok=True
)


# ============================================================
# STORAGE INITIALIZATION
# ============================================================

def initialize_storage():

    if not os.path.exists(
        EMPLOYEES_FILE
    ):

        with open(
            EMPLOYEES_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                {},
                file,
                indent=4
            )

    if not os.path.exists(
        ATTENDANCE_FILE
    ):

        df = pd.DataFrame(
            columns=[
                "Employee ID",
                "Employee Name",
                "Date",
                "Event",
                "Time"
            ]
        )

        df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )


initialize_storage()


# ============================================================
# EMPLOYEE FUNCTIONS
# ============================================================

def load_employees():

    try:

        with open(
            EMPLOYEES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:

        return {}


def save_employees(
    employees
):

    with open(
        EMPLOYEES_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            employees,
            file,
            indent=4
        )


# ============================================================
# ATTENDANCE FUNCTIONS
# ============================================================

def load_attendance():

    try:

        df = pd.read_csv(
            ATTENDANCE_FILE
        )

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError
    ):

        df = pd.DataFrame()

    required_columns = [
        "Employee ID",
        "Employee Name",
        "Date",
        "Event",
        "Time"
    ]

    if (
        df.empty
        or not all(
            column in df.columns
            for column in required_columns
        )
    ):

        df = pd.DataFrame(
            columns=required_columns
        )

    return df


# ============================================================
# FACE EMBEDDING
# ============================================================

def generate_embedding(
    image_path
):

    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True
    )

    if not result:

        raise ValueError(
            "No face detected."
        )

    if len(result) != 1:

        raise ValueError(
            "Please keep only one face in front of the camera."
        )

    return np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )


# ============================================================
# COSINE DISTANCE
# ============================================================

def cosine_distance(
    embedding_a,
    embedding_b
):

    a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:

        return 1.0

    similarity = np.dot(
        a,
        b
    ) / (
        norm_a * norm_b
    )

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    return float(
        1.0 - similarity
    )


# ============================================================
# EMPLOYEE RECOGNITION
# ============================================================

def recognize_employee(
    query_embedding
):

    employees = load_employees()

    best_employee = None
    best_distance = float("inf")

    for employee_id, employee_data in employees.items():

        if "embedding" not in employee_data:
            continue

        try:

            stored_embedding = np.asarray(
                employee_data["embedding"],
                dtype=np.float32
            )

            distance = cosine_distance(
                query_embedding,
                stored_embedding
            )

            if distance < best_distance:

                best_distance = distance
                best_employee = employee_id

        except Exception:

            continue

    if (
        best_employee
        and best_distance <= MATCH_THRESHOLD
    ):

        return (
            best_employee,
            best_distance
        )

    return (
        None,
        best_distance
    )


# ============================================================
# NEXT EVENT
# ============================================================

def get_next_event(
    employee_id,
    current_date
):

    attendance = load_attendance()

    records = attendance[
        (attendance["Employee ID"] == employee_id)
        &
        (attendance["Date"] == current_date)
    ]

    if records.empty:

        return "CHECK-IN"

    last_event = str(
        records.iloc[-1]["Event"]
    )

    if last_event == "CHECK-IN":

        return "CHECK-OUT"

    return "CHECK-IN"


# ============================================================
# COOLDOWN
# ============================================================

def check_cooldown(
    employee_id,
    current_date,
    current_datetime
):

    attendance = load_attendance()

    records = attendance[
        (attendance["Employee ID"] == employee_id)
        &
        (attendance["Date"] == current_date)
    ]

    if records.empty:

        return False, 0

    last_record = records.iloc[-1]

    try:

        last_datetime = datetime.strptime(
            f"{current_date} {last_record['Time']}",
            "%Y-%m-%d %H:%M:%S"
        )

    except Exception:

        return False, 0

    elapsed = (
        current_datetime - last_datetime
    ).total_seconds()

    if elapsed < SCAN_COOLDOWN_SECONDS:

        remaining = int(
            SCAN_COOLDOWN_SECONDS - elapsed
        )

        return True, remaining

    return False, 0


# ============================================================
# RECORD ATTENDANCE
# ============================================================

def record_event(
    employee_id
):

    employees = load_employees()
    attendance = load_attendance()

    employee_name = employees[
        employee_id
    ]["name"]

    now = datetime.now()

    current_date = now.strftime(
        "%Y-%m-%d"
    )

    current_time = now.strftime(
        "%H:%M:%S"
    )

    cooldown, remaining = check_cooldown(
        employee_id,
        current_date,
        now
    )

    if cooldown:

        return {
            "recorded": False,
            "cooldown": True,
            "remaining": remaining
        }

    event = get_next_event(
        employee_id,
        current_date
    )

    new_record = pd.DataFrame(
        {
            "Employee ID": [
                employee_id
            ],
            "Employee Name": [
                employee_name
            ],
            "Date": [
                current_date
            ],
            "Event": [
                event
            ],
            "Time": [
                current_time
            ]
        }
    )

    attendance = pd.concat(
        [
            attendance,
            new_record
        ],
        ignore_index=True
    )

    attendance.to_csv(
        ATTENDANCE_FILE,
        index=False
    )

    return {
        "recorded": True,
        "cooldown": False,
        "employee_id": employee_id,
        "employee_name": employee_name,
        "event": event,
        "date": current_date,
        "time": current_time
    }


# ============================================================
# DAILY SUMMARY
# ============================================================

def calculate_daily_summary(
    employee_id,
    date
):

    attendance = load_attendance()

    records = attendance[
        (attendance["Employee ID"] == employee_id)
        &
        (attendance["Date"] == date)
    ].reset_index(
        drop=True
    )

    if records.empty:

        return {
            "first_checkin": "-",
            "last_checkout": "-",
            "visits": 0,
            "working_seconds": 0,
            "status": "OUT"
        }

    total_seconds = 0

    visits = 0

    first_checkin = None
    last_checkout = None

    current_checkin = None

    for _, row in records.iterrows():

        event = str(
            row["Event"]
        )

        time_string = str(
            row["Time"]
        )

        try:

            event_time = datetime.strptime(
                f"{date} {time_string}",
                "%Y-%m-%d %H:%M:%S"
            )

        except Exception:

            continue

        if event == "CHECK-IN":

            if first_checkin is None:

                first_checkin = event_time

            current_checkin = event_time

        elif (
            event == "CHECK-OUT"
            and current_checkin is not None
        ):

            duration = (
                event_time
                - current_checkin
            ).total_seconds()

            if duration >= 0:

                total_seconds += duration
                visits += 1

            last_checkout = event_time

            current_checkin = None

    status = (
        "IN"
        if current_checkin is not None
        else "OUT"
    )

    return {
        "first_checkin": (
            first_checkin.strftime(
                "%H:%M:%S"
            )
            if first_checkin
            else "-"
        ),
        "last_checkout": (
            last_checkout.strftime(
                "%H:%M:%S"
            )
            if last_checkout
            else "-"
        ),
        "visits": visits,
        "working_seconds": int(
            total_seconds
        ),
        "status": status
    }


# ============================================================
# FORMAT DURATION
# ============================================================

def format_duration(
    seconds
):

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# LIVE CAMERA PROCESSOR
# ============================================================

class FaceCameraProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.frame_count = 0

        self.latest_frame = None

        self.lock = None

    def recv(
        self,
        frame
    ):

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        # Store latest frame
        if (
            self.frame_count
            % PROCESS_EVERY_N_FRAMES
            == 0
        ):

            self.latest_frame = image.copy()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">'
    'Employee Attendance System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Live face verification and automated employee attendance'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### System"
    )

    st.markdown(
        '<p class="status-online">'
        'System Online'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.write(
        f"Version: {APP_VERSION}"
    )

    st.write(
        f"Recognition Model: {MODEL_NAME}"
    )

    st.write(
        f"Face Detector: {DETECTOR_BACKEND}"
    )

    st.write(
        f"Match Threshold: {MATCH_THRESHOLD}"
    )

    st.write(
        f"Scan Cooldown: "
        f"{SCAN_COOLDOWN_SECONDS} seconds"
    )

    st.divider()

    st.caption(
        "Prototype environment"
    )


# ============================================================
# DASHBOARD
# ============================================================

employees = load_employees()

attendance = load_attendance()

today = datetime.now().strftime(
    "%Y-%m-%d"
)

today_records = attendance[
    attendance["Date"] == today
]

checked_in_ids = set(
    today_records[
        today_records["Event"] == "CHECK-IN"
    ]["Employee ID"]
    .tolist()
)

checked_out_ids = set(
    today_records[
        today_records["Event"] == "CHECK-OUT"
    ]["Employee ID"]
    .tolist()
)

active_ids = (
    checked_in_ids
    - checked_out_ids
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Registered Employees",
        len(employees)
    )

with c2:

    st.metric(
        "Today's Events",
        len(today_records)
    )

with c3:

    st.metric(
        "Currently Inside",
        len(active_ids)
    )

with c4:

    st.metric(
        "System Status",
        "Online"
    )


st.divider()


# ============================================================
# TABS
# ============================================================

attendance_tab, registration_tab, records_tab = st.tabs(
    [
        "Live Attendance",
        "Employee Registration",
        "Attendance Records"
    ]
)


# ============================================================
# LIVE ATTENDANCE
# ============================================================

with attendance_tab:

    st.markdown(
        '<div class="section-title">'
        'Live Employee Verification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Start the camera. The system automatically detects '
        'and verifies employees without requiring a capture button.'
        '</div>',
        unsafe_allow_html=True
    )

    if not employees:

        st.warning(
            "No employees are registered. "
            "Register an employee first."
        )

    else:

        # Refresh the Streamlit page periodically
        st_autorefresh(
            interval=1000,
            key="attendance_refresh"
        )

        webrtc_ctx = webrtc_streamer(
            key="employee-attendance-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FaceCameraProcessor,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480
                },
                "audio": False
            },
            async_processing=True
        )

        st.info(
            "Position one employee in front of the camera. "
            "Keep the face clearly visible."
        )

        # ----------------------------------------
        # PROCESS LATEST FRAME
        # ----------------------------------------

        if (
            webrtc_ctx.state.playing
            and webrtc_ctx.video_processor
        ):

            processor = (
                webrtc_ctx.video_processor
            )

            frame = (
                processor.latest_frame
            )

            if frame is not None:

                temp_path = None

                try:

                    # Resize for faster processing
                    height, width = frame.shape[:2]

                    max_width = 640

                    if width > max_width:

                        scale = (
                            max_width / width
                        )

                        frame = cv2.resize(
                            frame,
                            (
                                int(width * scale),
                                int(height * scale)
                            )
                        )

                    # Save current frame
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp_file:

                        cv2.imwrite(
                            temp_file.name,
                            frame
                        )

                        temp_path = (
                            temp_file.name
                        )

                    with st.spinner(
                        "Verifying face..."
                    ):

                        embedding = (
                            generate_embedding(
                                temp_path
                            )
                        )

                        employee_id, distance = (
                            recognize_employee(
                                embedding
                            )
                        )

                    if employee_id:

                        result = record_event(
                            employee_id
                        )

                        if result["cooldown"]:

                            st.warning(
                                "Employee already scanned. "
                                f"Please wait "
                                f"{result['remaining']} seconds."
                            )

                        elif result["recorded"]:

                            if (
                                result["event"]
                                == "CHECK-IN"
                            ):

                                st.success(
                                    "Check-in recorded successfully."
                                )

                            else:

                                st.success(
                                    "Check-out recorded successfully."
                                )

                            x1, x2, x3, x4 = (
                                st.columns(4)
                            )

                            with x1:

                                st.metric(
                                    "Employee ID",
                                    result["employee_id"]
                                )

                            with x2:

                                st.metric(
                                    "Employee",
                                    result["employee_name"]
                                )

                            with x3:

                                st.metric(
                                    "Event",
                                    result["event"]
                                )

                            with x4:

                                st.metric(
                                    "Time",
                                    result["time"]
                                )

                            summary = (
                                calculate_daily_summary(
                                    employee_id,
                                    result["date"]
                                )
                            )

                            st.divider()

                            st.markdown(
                                "#### Today's Summary"
                            )

                            s1, s2, s3, s4 = (
                                st.columns(4)
                            )

                            with s1:

                                st.metric(
                                    "First Check-in",
                                    summary[
                                        "first_checkin"
                                    ]
                                )

                            with s2:

                                st.metric(
                                    "Last Check-out",
                                    summary[
                                        "last_checkout"
                                    ]
                                )

                            with s3:

                                st.metric(
                                    "Office Visits",
                                    summary[
                                        "visits"
                                    ]
                                )

                            with s4:

                                st.metric(
                                    "Working Time",
                                    format_duration(
                                        summary[
                                            "working_seconds"
                                        ]
                                    )
                                )

                            st.caption(
                                f"Recognition distance: "
                                f"{distance:.4f}"
                            )

                    else:

                        st.warning(
                            "Face detected, but no registered "
                            "employee matched."
                        )

                        if distance != float("inf"):

                            st.caption(
                                f"Best distance: "
                                f"{distance:.4f}"
                            )

                except ValueError as error:

                    st.warning(
                        f"Face processing: {error}"
                    )

                except Exception as error:

                    st.error(
                        "Verification error."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(error)
                        )

                finally:

                    if (
                        temp_path
                        and os.path.exists(
                            temp_path
                        )
                    ):

                        try:

                            os.remove(
                                temp_path
                            )

                        except Exception:

                            pass


# ============================================================
# EMPLOYEE REGISTRATION
# ============================================================

with registration_tab:

    st.markdown(
        '<div class="section-title">'
        'Employee Registration'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Register an employee once. The generated facial '
        'template will be used for automatic verification.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 1]
    )

    with left:

        new_employee_id = st.text_input(
            "Employee ID",
            placeholder="EMP002"
        )

        new_employee_name = st.text_input(
            "Employee Name",
            placeholder="Employee full name"
        )

    with right:

        registration_image = st.camera_input(
            "Registration Photo",
            key="employee_registration"
        )

    if st.button(
        "Register Employee",
        type="primary",
        use_container_width=True
    ):

        employee_id = (
            new_employee_id
            .strip()
            .upper()
        )

        employee_name = (
            new_employee_name
            .strip()
        )

        if not employee_id:

            st.error(
                "Employee ID is required."
            )

        elif not employee_name:

            st.error(
                "Employee Name is required."
            )

        elif registration_image is None:

            st.error(
                "Registration photo is required."
            )

        else:

            employees = load_employees()

            if employee_id in employees:

                st.error(
                    f"Employee {employee_id} "
                    "is already registered."
                )

            else:

                temp_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False
                    ) as temp_file:

                        temp_file.write(
                            registration_image.getvalue()
                        )

                        temp_path = (
                            temp_file.name
                        )

                    with st.spinner(
                        "Creating facial template..."
                    ):

                        embedding = (
                            generate_embedding(
                                temp_path
                            )
                        )

                    employees[employee_id] = {
                        "name": employee_name,
                        "embedding": embedding.tolist(),
                        "registered_at": (
                            datetime.now()
                            .isoformat()
                        )
                    }

                    save_employees(
                        employees
                    )

                    st.success(
                        f"Employee {employee_id} "
                        "registered successfully."
                    )

                except ValueError as error:

                    st.error(
                        f"Registration failed: {error}"
                    )

                except Exception as error:

                    st.error(
                        "Registration failed."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(error)
                        )

                finally:

                    if (
                        temp_path
                        and os.path.exists(
                            temp_path
                        )
                    ):

                        try:

                            os.remove(
                                temp_path
                            )

                        except Exception:

                            pass


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

with records_tab:

    st.markdown(
        '<div class="section-title">'
        'Attendance Records'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Review employee entry and exit history.'
        '</div>',
        unsafe_allow_html=True
    )

    records = load_attendance()

    if records.empty:

        st.info(
            "No attendance records available."
        )

    else:

        employee_options = [
            "All Employees"
        ] + sorted(
            records["Employee ID"]
            .dropna()
            .unique()
            .tolist()
        )

        f1, f2 = st.columns(2)

        with f1:

            selected_employee = st.selectbox(
                "Employee",
                employee_options
            )

        with f2:

            selected_date = st.date_input(
                "Date",
                value=datetime.now().date()
            )

        filtered = records.copy()

        if selected_employee != "All Employees":

            filtered = filtered[
                filtered["Employee ID"]
                == selected_employee
            ]

        date_string = selected_date.strftime(
            "%Y-%m-%d"
        )

        filtered = filtered[
            filtered["Date"] == date_string
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        if selected_employee != "All Employees":

            summary = calculate_daily_summary(
                selected_employee,
                date_string
            )

            st.divider()

            st.markdown(
                "#### Daily Summary"
            )

            a1, a2, a3, a4, a5 = (
                st.columns(5)
            )

            with a1:

                st.metric(
                    "First Check-in",
                    summary["first_checkin"]
                )

            with a2:

                st.metric(
                    "Last Check-out",
                    summary["last_checkout"]
                )

            with a3:

                st.metric(
                    "Visits",
                    summary["visits"]
                )

            with a4:

                st.metric(
                    "Working Time",
                    format_duration(
                        summary[
                            "working_seconds"
                        ]
                    )
                )

            with a5:

                st.metric(
                    "Current Status",
                    summary["status"]
                )

        st.download_button(
            "Download Attendance Report",
            data=filtered.to_csv(
                index=False
            ),
            file_name="attendance_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"{APP_NAME} | Prototype Version {APP_VERSION}"
)