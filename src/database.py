import os
import sqlite3


DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "lucent.db"
)


def get_connection():
    """
    Create and return a connection to the LUCENT SQLite database.
    """

    os.makedirs(
        os.path.dirname(DATABASE_PATH),
        exist_ok=True
    )

    connection = sqlite3.connect(DATABASE_PATH)

    return connection


def create_tables():
    """
    Create the required LUCENT database tables.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_duration REAL,
            blink_rate REAL,
            avg_blink_duration REAL,
            avg_blink_interval REAL,
            face_size REAL,
            state TEXT,
            wellness_score INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            alert_type TEXT,
            message TEXT,
            FOREIGN KEY (session_id)
                REFERENCES sessions(id)
        )
    """)

    connection.commit()

    connection.close()


def save_session(
    timestamp,
    session_duration,
    blink_rate,
    avg_blink_duration,
    avg_blink_interval,
    face_size,
    state,
    wellness_score
):
    """
    Save a completed monitoring session.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO sessions (
            timestamp,
            session_duration,
            blink_rate,
            avg_blink_duration,
            avg_blink_interval,
            face_size,
            state,
            wellness_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        session_duration,
        blink_rate,
        avg_blink_duration,
        avg_blink_interval,
        face_size,
        state,
        wellness_score
    ))

    session_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return session_id


def save_alert(
    session_id,
    timestamp,
    alert_type,
    message
):
    """
    Save an alert generated during a session.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO alerts (
            session_id,
            timestamp,
            alert_type,
            message
        )
        VALUES (?, ?, ?, ?)
    """, (
        session_id,
        timestamp,
        alert_type,
        message
    ))

    connection.commit()

    connection.close()

def get_recent_sessions(limit=10):
    """
    Retrieve recent monitoring sessions.
    """

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            timestamp,
            session_duration,
            blink_rate,
            avg_blink_duration,
            avg_blink_interval,
            face_size,
            state,
            wellness_score
        FROM sessions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    sessions = cursor.fetchall()

    connection.close()

    return sessions    