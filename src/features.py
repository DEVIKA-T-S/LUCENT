import math


def distance(point1, point2):
    """
    Calculate the distance between two facial landmark points.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
    )


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).
    """

    horizontal = distance(
        eye_points[0],
        eye_points[3]
    )

    vertical_1 = distance(
        eye_points[1],
        eye_points[5]
    )

    vertical_2 = distance(
        eye_points[2],
        eye_points[4]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


def calculate_face_size(face_landmarks):
    """
    Calculate the relative size of the detected face.

    Larger value = face appears closer.
    Smaller value = face appears farther.
    """

    min_x = min(
        point.x for point in face_landmarks
    )

    max_x = max(
        point.x for point in face_landmarks
    )

    min_y = min(
        point.y for point in face_landmarks
    )

    max_y = max(
        point.y for point in face_landmarks
    )

    face_width = max_x - min_x
    face_height = max_y - min_y

    face_size = (
        face_width + face_height
    ) / 2

    return face_size