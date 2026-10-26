"""
MediaPipe Face Mesh landmark indices (478 points) + custom landmark IDs.

Custom landmarks are defined in custom_landmark.py as CUSTOM_LANDMARK_DEFS.
Their logical IDs double as array indices in the extended coordinates array
(see core/landmark_resolver.py).
"""

from .custom_landmark import EDGE_LANDMARK_DIRECTIONS  # noqa: F401 — re-exported for eyeliner corrector

# ---------------------------------------------------------------------------
# Custom landmark IDs (logical ID = array index)
# ---------------------------------------------------------------------------
LM_1001 = 1001
LM_1002 = 1002
LM_1003 = 1003

LM_1101 = 1101
LM_1102 = 1102
LM_1103 = 1103
LM_1104 = 1104
LM_1105 = 1105

LM_1401 = 1401
LM_1402 = 1402
LM_1403 = 1403
LM_1404 = 1404
LM_1405 = 1405
LM_1406 = 1406
LM_1407 = 1407

LM_1501 = 1501
LM_1502 = 1502
LM_1503 = 1503
LM_1504 = 1504
LM_1505 = 1505
LM_1506 = 1506
LM_1507 = 1507

LM_1601 = 1601
LM_1602 = 1602
LM_1603 = 1603
LM_1604 = 1604
LM_1605 = 1605
LM_1606 = 1606
LM_1607 = 1607

LM_1701 = 1701
LM_1702 = 1702
LM_1703 = 1703
LM_1704 = 1704
LM_1705 = 1705
LM_1706 = 1706
LM_1707 = 1707

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
# Upper and lower lip indices; combined via bitwise OR in passes/lips.py
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

# Eyelid - area for eye shadow.
# The band is bounded by the crease line (top) and the lash line (bottom).
# EyeShadowMaskStrategy lerps the upper boundary from the lash line (size=0)
# toward the crease line (size=1), so a small size paints a thin band along
# the lash line and 1.0 covers the whole lid.
EYELID_CREASE_LEFT_INDICES = [113, 225, 224, 223, 222, 221, 190]  # outer->inner
EYELID_LASH_LEFT_INDICES = [243, 133, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]  # inner->outer

EYELID_CREASE_RIGHT_INDICES = [342, 445, 444, 443, 442, 441, 414]  # outer->inner
EYELID_LASH_RIGHT_INDICES = [463, 362, 398, 384, 385, 386, 387, 388, 466, 263, 359, 446]  # inner->outer

# Full closed polygon (crease outer->inner + lash inner->outer), kept for
# consumers that need the concatenated contour.
EYELID_LEFT_INDICES = EYELID_CREASE_LEFT_INDICES + EYELID_LASH_LEFT_INDICES
EYELID_RIGHT_INDICES = EYELID_CREASE_RIGHT_INDICES + EYELID_LASH_RIGHT_INDICES

# Eyelid zones - horizontal thirds (outer/center/inner) of the eyelid band.
# Each zone is a (crease outer->inner, lash inner->outer) pair. Adjacent zones
# share their crease seam vertices; the lash lists overlap by one point into
# the neighbouring zone so the zone boundary edges stagger rather than aligning
# into a single visible kink.
EYELID_OUTER_LEFT_INDICES = ([113, 225, 224], [159, 160, 161, 246, 33, 130, 226])
EYELID_CENTER_LEFT_INDICES = ([224, 223, 222], [157, 158, 159, 160, 161])
EYELID_INNER_LEFT_INDICES = ([222, 221, 190], [243, 133, 173, 157, 158, 159])

EYELID_OUTER_RIGHT_INDICES = ([342, 445, 444], [386, 387, 388, 466, 263, 359, 446])
EYELID_CENTER_RIGHT_INDICES = ([444, 443, 442], [384, 385, 386, 387, 388])
EYELID_INNER_RIGHT_INDICES = ([442, 441, 414], [463, 362, 398, 384, 385, 386])

# Extended zones — adjacent zone pairs merged into single (crease, lash) tuples.
# inner_center = inner ∪ center,  outer_center = outer ∪ center. The lash lists
# carry the same one-point overlap into the neighbouring third as the base zones.
EYELID_INNER_CENTER_LEFT_INDICES = ([224, 223, 222, 221, 190], [243, 133, 173, 157, 158, 159, 160, 161])
EYELID_OUTER_CENTER_LEFT_INDICES = ([113, 225, 224, 223, 222], [157, 158, 159, 160, 161, 246, 33, 130, 226])
EYELID_INNER_CENTER_RIGHT_INDICES = ([444, 443, 442, 441, 414], [463, 362, 398, 384, 385, 386, 387, 388])
EYELID_OUTER_CENTER_RIGHT_INDICES = ([342, 445, 444, 443, 442], [384, 385, 386, 387, 388, 466, 263, 359, 446])


EYELINER_UPPER_LEFT_INDICES = [
    LM_1507, LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407, 130, 226
]

EYELINER_LOWER_LEFT_INDICES = [
    130, LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507, 133
]

EYELINER_UPPER_RIGHT_INDICES = [
    LM_1707, LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607, 359, 446
]

EYELINER_LOWER_RIGHT_INDICES = [
    359, LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707, 362,
]


TEAR_BAG_SHADOW_LEFT_INDICES = [233, 232, 231, 230, 229, 228, 31]
TEAR_BAG_SHADOW_RIGHT_INDICES = [453, 452, 451, 450, 449, 448, 261]

TEAR_BAG_ROOT_LEFT_INDICES = [LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507]
TEAR_BAG_ROOT_RIGHT_INDICES = [LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707]

TEAR_BAG_ENVELOPE = (0.1, 0.45, 0.5, 0.45, 0.35, 0.25, 0.1)

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
LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

EYE_IRIS_RIGHT_INDICES = [474, 475, 476, 477, 473]
EYE_IRIS_LEFT_INDICES = [469, 470, 471, 472, 468]

# Face oval contour (jawline + forehead) for foundation mask
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


