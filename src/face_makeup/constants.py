"""
MediaPipe Face Mesh landmark indices (478 points).

Custom landmarks are defined in custom_landmark.py.
Use resolve_indices() to expand index lists (including 'mid' tuples) to coordinates.
"""

from .custom_landmark import (
    LM_1001, LM_1002, LM_1003,
    LM_1101, LM_1102, LM_1103, LM_1104, LM_1105,
    LM_1201, LM_1202, LM_1203, LM_1204, LM_1205, LM_1206, LM_1207,
    LM_1301, LM_1302, LM_1303, LM_1304, LM_1305, LM_1306, LM_1307,
    LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407, LM_1408,
    LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
    LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607, LM_1608,
    LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
    resolve_indices,
)

# MediaPipe Face Mesh landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md


# Lips (upper lip)
UPPER_LIP_INDICES = [
    185, 39, LM_1001, LM_1002, LM_1003, 270, 409,
    272, 271, 268, 12, 38, 41, 42,
]


# Lower lip - outer contour only
LOWER_LIP_INDICES = [
    89, LM_1101, LM_1102, LM_1103, LM_1104, LM_1105, 319,
    307, 320, 404, 315, 16, 85, 180, 90, 77,
]


# Full lips (upper + lower combined)
# Note: Processed separately in segmenter.py then combined
LIPS_INDICES = {
    'upper': UPPER_LIP_INDICES,
    'lower': LOWER_LIP_INDICES,
}

# Left cheek
# Area below the cheekbone
CHEEK_LEFT_INDICES = [
    123, 192, 203, 205, 206, 207, 36, 212, 213, 214,
    142, 147, 116, 117, 119, 50, 101, 118, 187,
]

# Right cheek
# Area below the cheekbone
CHEEK_RIGHT_INDICES = [
    371, 376, 266, 280, 411, 416, 423, 425, 426, 427,
    432, 433, 434, 330, 345, 346, 347, 348, 352,
]

# Eyelid - area for eye shadow
EYELID_LEFT_INDICES = [
    113, 225, 224, 223, 222, 221, 190,
    243, 133, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226,
]

EYELID_RIGHT_INDICES = [
    342, 445, 444, 443, 442, 441, 414,
    463, 362, 398, 384, 385, 386, 387, 388, 466, 263, 359, 446,
]

# Eyelid edge - crease area
EYELID_EDGE_LEFT_INDICES = [
    LM_1201, LM_1202, LM_1203, LM_1204, LM_1205, LM_1206, LM_1207,
    243, 133, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226,
]

EYELID_EDGE_RIGHT_INDICES = [
    LM_1301, LM_1302, LM_1303, LM_1304, LM_1305, LM_1306, LM_1307,
    463, 362, 398, 384, 385, 386, 387, 388, 466, 263, 359, 446,
]

EYELINER_UPPER_LEFT_INDICES = [
    LM_1507, LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407, LM_1408, 130, 226
]

EYELINER_LOWER_LEFT_INDICES = [
    130, LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507, 133
]

EYELINER_UPPER_RIGHT_INDICES = [
    LM_1707, LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607, LM_1608, 359, 446
]

EYELINER_LOWER_RIGHT_INDICES = [
    359, LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707, 362,
]


EYELINER_LEFT_INDICES = {
    'upper': EYELINER_UPPER_LEFT_INDICES,
    'lower': EYELINER_LOWER_LEFT_INDICES,
}

EYELINER_RIGHT_INDICES = {
    'upper': EYELINER_UPPER_RIGHT_INDICES,
    'lower': EYELINER_LOWER_RIGHT_INDICES,
}

EYELINER_CORRECTION_MAP = {
    'eyeliner_left': {
        'upper': EYELINER_UPPER_LEFT_INDICES,
        'lower': EYELINER_LOWER_LEFT_INDICES,
    },
    'eyeliner_right': {
        'upper': EYELINER_UPPER_RIGHT_INDICES,
        'lower': EYELINER_LOWER_RIGHT_INDICES,
    },
}



# Eye opening contour - used to clip the iris ellipse mask so contact-lens
# color never bleeds onto eyelid skin when the eye is half-closed (the
# fitEllipse iris mask alone can overlap the eyelid in that case).
LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

EYE_IRIS_RIGHT_INDICES = [474, 475, 476, 477, 473]
EYE_IRIS_LEFT_INDICES = [469, 470, 471, 472, 468]


# Part name to landmark indices mapping
# Format: 'part_name': INDICES or 'part_name': (INDICES, 'type')
PART_LANDMARK_MAP = {
    # Convex hull type
    'cheek_left': (CHEEK_LEFT_INDICES, "convex"),
    'cheek_right': (CHEEK_RIGHT_INDICES, "convex"),

    # Polygon type
    'lips': (LIPS_INDICES, 'polygon'),
    'eye_left': (LEFT_EYE_IDX, 'polygon'),
    'eye_right': (RIGHT_EYE_IDX, 'polygon'),
    'eyelid_left': (EYELID_LEFT_INDICES, 'polygon'),
    'eyelid_right': (EYELID_RIGHT_INDICES, 'polygon'),
    'eyelid_edge_left': (EYELID_EDGE_LEFT_INDICES, 'polygon'),
    'eyelid_edge_right': (EYELID_EDGE_RIGHT_INDICES, 'polygon'),

    # Curve type
    'eyeliner_left': (EYELINER_LEFT_INDICES, 'curve'),
    'eyeliner_right': (EYELINER_RIGHT_INDICES, 'curve'),

    # Ellipse type
    'eyeiris_left': (EYE_IRIS_LEFT_INDICES, 'ellipse'),
    'eyeiris_right': (EYE_IRIS_RIGHT_INDICES, 'ellipse'),
}
