"""
MediaPipe Face Mesh landmark indices (478点)

中点（Midpoint）機能:
  ランドマークインデックスとして、2点以上の中点・重心を指定できます。

  形式: ('mid', idx1, idx2, ..., idxN)

  例:
    ('mid', 33, 133)           # 点33と点133の中点（2点の平均）
    ('mid', 33, 133, 160)      # 点33, 133, 160の重心（3点の平均）

  使用方法:
    EXAMPLE_INDICES = [
        33,                    # 通常のランドマーク
        ('mid', 33, 133),      # 中点（仮想ランドマーク）
        159,                   # 通常のランドマーク
    ]

  注意:
    - 中点は実行時に計算されます
    - 中点インデックスは既存の478点に含まれません
"""

# MediaPipe Face Mesh landmark indices
# 参考: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md

# 唇（Lips）
UPPER_LIP_INDICES = [
    185, 74, 73, 72, 11, 302, 303, 304, 408,
         42, 41, 38, 12, 268, 271, 272,
]

# 下唇（Lower Lip）- 外側の輪郭のみ
LOWER_LIP_INDICES = [
         
         319, 403, 316, 15, 86, 179, 89,
    307, 320, 404, 315, 16, 85, 180, 90, 146,
]

# 唇全体（上唇+下唇を結合）
# 注意: segmenter.pyで別々に処理してから結合する
LIPS_INDICES = {
    'upper': UPPER_LIP_INDICES,
    'lower': LOWER_LIP_INDICES,
}

# 左頬（Cheek Left）
# 頬骨下あたりの領域
CHEEK_LEFT_INDICES = [123, 192, 203, 205, 206, 207, 36, 212, 213, 214, 142, 147, 116, 117, 119,50,101,118,187]

# 右頬（Cheek Right）
# 頬骨下あたりの領域
CHEEK_RIGHT_INDICES = [371, 376, 266, 280, 411, 416, 423, 425, 426, 427, 432, 433, 434, 330, 345, 346, 347, 348, 352]

# まぶた（Eyelid）- アイシャドウ用の領域
EYELID_LEFT_INDICES = [
        113,225,224,223,222,221,190,
    243,133,173,157,158,159,160,161,246,33,130,226
]

EYELID_RIGHT_INDICES = [
        342, 445, 444, 443, 442, 441, 414,
    463,362,398,384,385,386,387,388,466,263,359,446]

# まぶたのエッジ（Eyelid Edge）- 目の際
EYELID_EDGE_LEFT_INDICES = [

    ("mid",247,33),
    ("mid",30,161),
    ("mid",29,160),
    ("mid",27,159),
    ("mid",28,158),
    ("mid",56,157),
    ("mid",190,243),

    243,133,173,157,158,159,160,161,246,33,130,226
]

EYELID_EDGE_RIGHT_INDICES = [

    ("mid",467,263),
    ("mid",260,388),
    ("mid",259,387),
    ("mid",257,386),
    ("mid",258,385),
    ("mid",286,384),
    ("mid",414,463),
    
    463,362,398,384,385,386,387,388,466,263,359,446]

EYELINER_UPPER_LEFT_INDICES=[
    243,133,173,157,158,159,160,161,246,33,130
]
EYELINER_LOWER_LEFT_INDICES=[
    33,7,163,144,145,153,154,155,133,243
]
EYELINER_UPPER_RIGHT_INDICES=[
    463,362,398,384,385,386,387,388,466,263,359
]
EYELINER_LOWER_RIGHT_INDICES=[
    263,249,390,373,374,380,381,382,362,463
]

EYELINER_LEFT_INDICES = {
    'upper': EYELINER_UPPER_LEFT_INDICES,
    'lower': EYELINER_LOWER_LEFT_INDICES,
}

EYELINER_RIGHT_INDICES = {
    'upper': EYELINER_UPPER_RIGHT_INDICES,
    'lower': EYELINER_LOWER_RIGHT_INDICES,
}


# 部位名とインデックスのマッピング
# 形式: 'part_name': INDICES または 'part_name': (INDICES, 'type')
PART_LANDMARK_MAP = {
    #凸包タイプ
    'cheek_left': (CHEEK_LEFT_INDICES,"convex"),
    'cheek_right': (CHEEK_RIGHT_INDICES,"convex"),
    
    #多角形タイプ
    'lips': (LIPS_INDICES, 'polygon'),
    'eyelid_left': (EYELID_LEFT_INDICES, 'polygon'),
    'eyelid_right': (EYELID_RIGHT_INDICES, 'polygon'),
    'eyelid_edge_left': (EYELID_EDGE_LEFT_INDICES, 'polygon'),
    'eyelid_edge_right': (EYELID_EDGE_RIGHT_INDICES, 'polygon'),
    
    #曲線タイプ
    'eyeliner_left': (EYELINER_LEFT_INDICES, 'curve'),
    'eyeliner_right': (EYELINER_RIGHT_INDICES, 'curve'),

    
}

# その他の部位定義（将来使用予定）
EYE_OUTER_V_RIGHT_INDICES = [
    255, 261, 446, 342, 467, 359
]

EYE_OUTER_V_LEFT_INDICES = [
    25, 31, 226, 113, 247, 130
]

EYE_TEARBAG_LEFT_INDICES=[
    382, 381, 380, 374, 373, 390, 255,339,254,253,252,256
]

EYE_TEARBAG_RIGHT_INDICES=[
    7, 163, 144, 145, 153, 154, 26, 22, 23, 24, 110, 25
    ]
