Face Attendance APK Backend

FastAPI bridge between the Android APK and the existing DeepFace attendance engine.

Local run

py -3.10 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

Health check: http://127.0.0.1:8000/health

APK endpoint: POST http://<LAPTOP-IP>:8000/attendance/verify

For local testing, phone and laptop must be on the same Wi-Fi.

The Streamlit app remains separate. The APK calls FastAPI, not Streamlit.