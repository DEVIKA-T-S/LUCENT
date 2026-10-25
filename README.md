# LUCENT

## Personalized Screen-Time Eye Strain and Blink Monitoring System

LUCENT is a Python-based educational wellness monitoring prototype that uses a webcam to analyze blinking behaviour and estimate eye fatigue.

## Features

- Real-time webcam monitoring
- Eye Aspect Ratio (EAR) blink detection
- Machine-learning-based fatigue prediction
- Four states: Normal, Low Blinking, Mild Fatigue and High Fatigue
- Wellness score and configurable alerts
- SQLite session history
- Streamlit dashboard and before/after analysis

## Technologies

Python, OpenCV, MediaPipe, NumPy, Pandas, Scikit-learn, Streamlit and SQLite.

## Installation

Create a virtual environment and install dependencies:

    python -m venv venv
    python -m pip install -r requirements.txt

## Run Webcam Monitoring

From the LUCENT project folder:

    python -m src.live_prediction

Press Q to stop monitoring and save the session.

## Run Dashboard

    python -m streamlit run dashboard/app.py

Open http://localhost:8501 in a browser. The dashboard displays saved monitoring sessions; webcam monitoring runs separately.

## Note

LUCENT is an educational prototype, not a medical diagnostic system.