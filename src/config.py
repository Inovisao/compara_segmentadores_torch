#todas as classes do problema
ALL_CLASSES = [
    'car',
    'road',
    'mark',
    'building',
    'sidewalk',
    'tree',
    'pole',
    'sign',
    'person',
    'wall',
    'sky',
    'curb',
    'grass',
    'void'
]
#usado no treinamento para mapear as classes com os valores de pixels
LABEL_COLORS_LIST = [
    (0, 0,255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 153),
    (153, 0, 255),
    (0, 153, 255),
    (153, 255, 0),
    (255, 153, 0),
    (0, 255, 153),
    (0, 153, 153),
    (0, 0, 0)
]

VIS_LABEL_MAP = [
    (0, 0,255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 153),
    (153, 0, 255),
    (0, 153, 255),
    (153, 255, 0),
    (255, 153, 0),
    (0, 255, 153),
    (0, 153, 153),
    (0, 0, 0)
]