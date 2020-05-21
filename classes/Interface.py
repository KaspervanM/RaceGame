class Interface:
    __slots__ = [
        "rotations",
        "inputkeys",
        "key",
        "viewline",
        "xc",
        "yc",
        "LevelLayer_width",
        "LevelLayer_height",
        "pixels",
        "score",
        "carWidth",
        "speed",
        "modelindex",
        "model",
    ]

    def __init__(
        self,
        rotations,
        inputkeys,
        key,
        viewline,
        xc,
        yc,
        LevelLayer_width,
        LevelLayer_height,
        pixels,
        score,
        carWidth,
        speed,
        modelindex,
        model,
    ):
        self.rotations = rotations  # 0
        self.inputkeys = inputkeys  # [0,0,0,0]
        self.key = key  # [0,0,0,0]
        self.viewline = viewline  # []
        self.xc = xc  # 0
        self.yc = yc  # 0
        self.LevelLayer_width = LevelLayer_width  # 0
        self.LevelLayer_height = LevelLayer_height  # 0
        self.pixels = pixels  # []
        self.score = score  # 0
        self.carWidth = carWidth  # 0
        self.speed = speed
        self.modelindex = modelindex
        self.model = model

    def set_code(self):
        inputkeys = self.inputkeys
        key = self.key
        if inputkeys[0] == 1:  # left
            self.key[0] = 1
        elif key[0] == 1:
            self.key[0] = 0

        if inputkeys[1] == 1:  # right
            self.key[1] = 1
        elif key[1] == 1:
            self.key[1] = 0

        if inputkeys[2] == 1:  # up
            if key[3] != 1:
                self.key[2] = 1
        elif key[2] == 1:
            if key[3] != -2:
                self.key[2] = -1
            else:
                self.key[2] = 0

        if inputkeys[3] == 1:  # down
            if key[2] != 1:
                self.key[3] = 1
        elif key[3] == 1:
            if key[2] != -1:
                self.key[3] = -2
            else:
                self.key[3] = 0
