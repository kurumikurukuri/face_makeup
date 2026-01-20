"""
MediaPipe Face Mesh landmark indices (478 points).

Midpoint feature:
  You can specify midpoints or centroids of 2 or more landmarks as indices.

  Format: ('mid', idx1, idx2, ..., idxN)

  Examples:
    ('mid', 33, 133)           # Midpoint of points 33 and 133 (average of 2 points)
    ('mid', 33, 133, 160)      # Centroid of points 33, 133, 160 (average of 3 points)

  Usage:
    EXAMPLE_INDICES = [
        33,                    # Regular landmark
        ('mid', 33, 133),      # Midpoint (virtual landmark)
        159,                   # Regular landmark
    ]

  Notes:
    - Midpoints are computed at runtime
    - Midpoint indices are not included in the existing 478 points
"""

# MediaPipe Face Mesh landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md

# Lips (upper lip)
UPPER_LIP_INDICES = [
    185, 74, 73, 72, 11, 302, 303, 304,409, 272,271,268,12,38,41,42,
]

# Lower lip - outer contour only
LOWER_LIP_INDICES = [
89,179,86,15,316,403,319,307, 320, 404, 315, 16, 85, 180, 90, 77,
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
    ("mid", 247, 33),
    ("mid", 30, 161),
    ("mid", 29, 160),
    ("mid", 27, 159),
    ("mid", 28, 158),
    ("mid", 56, 157),
    ("mid", 190, 243),
    243, 133, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226,
]

EYELID_EDGE_RIGHT_INDICES = [
    ("mid", 467, 263),
    ("mid", 260, 388),
    ("mid", 259, 387),
    ("mid", 257, 386),
    ("mid", 258, 385),
    ("mid", 286, 384),
    ("mid", 414, 463),
    463, 362, 398, 384, 385, 386, 387, 388, 466, 263, 359, 446,
]

EYELINER_UPPER_LEFT_INDICES = [
    133, 173, 157, 158, 159, 160, 161, 246, 33, 130,
]
EYELINER_LOWER_LEFT_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
]
EYELINER_UPPER_RIGHT_INDICES = [
    362, 398, 384, 385, 386, 387, 388, 466, 263, 359,
]
EYELINER_LOWER_RIGHT_INDICES = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
]

EYELINER_LEFT_INDICES = {
    'upper': EYELINER_UPPER_LEFT_INDICES,
    'lower': EYELINER_LOWER_LEFT_INDICES,
}

EYELINER_RIGHT_INDICES = {
    'upper': EYELINER_UPPER_RIGHT_INDICES,
    'lower': EYELINER_LOWER_RIGHT_INDICES,
}

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
