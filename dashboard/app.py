import streamlit as st
import pandas as pd
import json
import os

from src.database import get_recent_sessions, create_tables

create_tables()


# ==========================================
# SETTINGS FILE
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SETTINGS_PATH = os.path.join(
    PROJECT_ROOT,
    "config",
    "settings.json"
)


def load_settings():

    if not os.path.exists(SETTINGS_PATH):

        return {
            "blink_reminder_minutes": 1,
            "enable_alerts": True,
            "alert_cooldown_seconds": 60,
            "alert_display_seconds": 5
        }

    with open(SETTINGS_PATH, "r") as file:
        return json.load(file)


def save_settings(settings):

    with open(SETTINGS_PATH, "w") as file:
        json.dump(
            settings,
            file,
            indent=4
        )


settings = load_settings()


# ==========================================
# LUCENT DASHBOARD
# ==========================================

st.set_page_config(
    page_title="LUCENT",
    page_icon="👁️",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("👁️ LUCENT")
st.subheader("Personalized Screen-Time Eye Strain & Blink Monitoring System")

st.write(
    "Monitor blink behavior, fatigue-related states, "
    "wellness score, and session activity."
)




# LUCENT_LIGHT_THEME
st.markdown("""
<style>
.stApp {
    background-color: #F4F8FD;
    color: #334155;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] * {
    color: #334155;
}

h1, h2, h3 {
    color: #17365D !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 22px;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(30, 64, 110, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
}

[data-testid="stMetricValue"] {
    color: #17365D;
    font-weight: 700;
}

.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF;
}

[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}

hr {
    border-color: #E2E8F0;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stMetric"] {
        padding: 14px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("LUCENT")
st.sidebar.write("Monitoring Dashboard")

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Select a page:",
    [
        "Dashboard",
        "Session History",
        "Before / After Analysis",
        "Settings"
    ]
)


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.header("Live Monitoring")

    sessions = get_recent_sessions(1)

    if sessions:

        latest_session = sessions[0]

        session_id = latest_session[0]
        timestamp = latest_session[1]
        session_duration = latest_session[2]
        blink_rate = latest_session[3]
        avg_blink_duration = latest_session[4]
        avg_blink_interval = latest_session[5]
        face_size = latest_session[6]
        state = latest_session[7]
        wellness_score = latest_session[8]

        minutes = int(session_duration // 60)
        seconds = int(session_duration % 60)

        session_time = f"{minutes:02d}:{seconds:02d}"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current State",
                state
            )

        with col2:
            st.metric(
                "Blink Rate",
                f"{blink_rate:.2f} / min"
            )

        with col3:
            st.metric(
                "Wellness Score",
                wellness_score
            )

        with col4:
            st.metric(
                "Session Time",
                session_time
            )

        st.divider()

        st.subheader("Latest Session Details")

        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            st.write(
                f"**Session ID:** {session_id}"
            )

            st.write(
                f"**Session Time:** {timestamp}"
            )

            st.write(
                f"**Face Size:** {face_size:.3f}"
            )

        with detail_col2:
            st.write(
                f"**Average Blink Duration:** "
                f"{avg_blink_duration:.3f} s"
            )

            st.write(
                f"**Average Blink Interval:** "
                f"{avg_blink_interval:.3f} s"
            )

        with detail_col3:
            if state == "Normal":
                st.success(
                    "Current state: Normal"
                )

            elif state == "Low Blinking":
                st.warning(
                    "Low blinking detected."
                )

            elif state == "Mild Fatigue":
                st.warning(
                    "Mild fatigue-related state detected."
                )

            elif state == "High Fatigue":
                st.error(
                    "High fatigue-related state detected."
                )

        st.divider()

        st.subheader("Monitoring Status")

        st.info(
            "Latest LUCENT monitoring session loaded successfully."
        )

    else:

        st.info(
            "No monitoring sessions found. "
            "Run LUCENT live prediction first."
        )


# ==========================================
# SESSION HISTORY
# ==========================================

elif page == "Session History":

    st.header("Session History")

    st.write(
        "Recent LUCENT monitoring sessions"
    )

    sessions = get_recent_sessions(10)

    if sessions:

        columns = [
            "ID",
            "Timestamp",
            "Session Duration",
            "Blink Rate",
            "Avg Blink Duration",
            "Avg Blink Interval",
            "Face Size",
            "State",
            "Wellness Score"
        ]

        df = pd.DataFrame(
            sessions,
            columns=columns
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info(
            "No monitoring sessions found."
        )
# ==========================================
# BEFORE / AFTER ANALYSIS
# ==========================================
elif page == "Before / After Analysis":

    st.header("Before / After Analysis")

    st.write(
        "Compare two LUCENT monitoring sessions "
        "using recorded session data."
    )

    sessions = get_recent_sessions(10)

    if len(sessions) >= 2:

        session_options = {}

        for session in sessions:
            session_id = session[0]
            timestamp = session[1]

            session_options[
                f"Session {session_id} - {timestamp}"
            ] = session

        options = list(session_options.keys())

        before_option = st.selectbox(
            "Select BEFORE session",
            options,
            index=1
        )

        after_option = st.selectbox(
            "Select AFTER session",
            options,
            index=0
        )

        before_session = session_options[
            before_option
        ]

        after_session = session_options[
            after_option
        ]

        if before_session[0] == after_session[0]:

            st.warning(
                "Please select two different sessions."
            )

        else:

            from src.analysis import compare_sessions

            result = compare_sessions(
                before_session,
                after_session
            )

            st.divider()

            st.subheader("Session Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Before:** Session {before_session[0]}"
                )

            with col2:
                st.write(
                    f"**After:** Session {after_session[0]}"
                )

            st.divider()

            metrics = [
                ("blink_rate", "Blink Rate", "/ min"),
                (
                    "avg_blink_duration",
                    "Average Blink Duration",
                    "s"
                ),
                (
                    "avg_blink_interval",
                    "Average Blink Interval",
                    "s"
                ),
                (
                    "wellness_score",
                    "Wellness Score",
                    ""
                )
            ]

            for key, label, unit in metrics:

                data = result[key]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        f"Before - {label}",
                        f"{data['before']:.2f} {unit}"
                    )

                with col2:
                    st.metric(
                        f"After - {label}",
                        f"{data['after']:.2f} {unit}"
                    )

                with col3:
                    st.metric(
                        "Change",
                        f"{data['change']:.2f}"
                    )

            st.divider()

            st.subheader("Detected States")

            state_col1, state_col2 = st.columns(2)

            with state_col1:
                st.info(
                    f"Before: {result['state_before']}"
                )

            with state_col2:
                st.info(
                    f"After: {result['state_after']}"
                )

    else:

        st.info(
            "At least two monitoring sessions "
            "are required for comparison."
        )


# ==========================================
# SETTINGS
# ==========================================

elif page == "Settings":

    st.header("Settings")

    st.write(
        "Customize LUCENT monitoring and alert behavior."
    )

    # ==========================================
    # BLINK REMINDER
    # ==========================================

    st.subheader("Blink Reminder")

    reminder = st.slider(
        "Reminder interval (minutes)",
        min_value=1,
        max_value=30,
        value=int(
            settings["blink_reminder_minutes"]
        )
    )

    # ==========================================
    # ALERT SETTINGS
    # ==========================================

    st.subheader("Alert Settings")

    enable_alerts = st.checkbox(
        "Enable fatigue alerts",
        value=settings["enable_alerts"]
    )

    cooldown = st.slider(
        "Alert cooldown (seconds)",
        min_value=10,
        max_value=300,
        value=int(
            settings["alert_cooldown_seconds"]
        )
    )

    display_duration = st.slider(
        "Alert display duration (seconds)",
        min_value=1,
        max_value=30,
        value=int(
            settings["alert_display_seconds"]
        )
    )

    # ==========================================
    # SAVE SETTINGS
    # ==========================================

    if st.button("Save Settings"):

        settings["blink_reminder_minutes"] = reminder
        settings["enable_alerts"] = enable_alerts
        settings["alert_cooldown_seconds"] = cooldown
        settings["alert_display_seconds"] = display_duration

        save_settings(settings)

        st.success(
            "LUCENT settings saved successfully!"
        )

    # ==========================================
    # CURRENT SETTINGS
    # ==========================================

    st.divider()

    st.subheader("Current Settings")

    st.write(
        f"**Blink Reminder:** "
        f"{settings['blink_reminder_minutes']} minute(s)"
    )

    st.write(
        f"**Fatigue Alerts:** "
        f"{'Enabled' if settings['enable_alerts'] else 'Disabled'}"
    )

    st.write(
        f"**Alert Cooldown:** "
        f"{settings['alert_cooldown_seconds']} seconds"
    )

    st.write(
        f"**Alert Display:** "
        f"{settings['alert_display_seconds']} seconds"
    )