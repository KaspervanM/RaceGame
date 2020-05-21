import sys
import time
from ast import literal_eval
from math import ceil
from os import mkdir
from os.path import exists
from shutil import rmtree

import pyglet
from keras.models import load_model
from numpy.random import randint
from numpy.random import seed
from numpy.random import shuffle

from classes.Cars import Car
from classes.NN import gen_mutant
from classes.NN import generate_random_NNev_model
from classes.NN import NNev
from classes.NN import NNHuman
from classes.NN import NNLongest  # skipcq: PYL-W0611

seed(10)


def get_platform():
    platforms = {
        "linux1": "Linux",
        "linux2": "Linux",
        "darwin": "OS X",
        "win32": "Windows",
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


fs = "\\" if get_platform() == "Windows" else "/"
currdir = sys.path[0]
car_image = pyglet.image.load(currdir + fs + "resources" + fs + "car.png")
Level = pyglet.image.load(currdir + fs + "resources" + fs + "Level.png").get_texture()
LevelLayer = pyglet.image.load(
    currdir + fs + "resources" + fs + "LevelLayerWscore2.png"
).get_texture()

b = True
show_cars = True
inf = False
showmodel = False

Human = 0

nModels = 3 * (1 - Human)
nCarsPerModel = 3 * (1 - Human)
nCars = nModels * nCarsPerModel + Human
car = []

generation = 0
start_time = int(time.time())
models = {}
model_scores = []

modelpath = currdir + fs + "resources" + fs + "models" + fs


def update(dt=0):
    dt = 0.03
    for i in range(nCars):
        if car[i]:
            car[i].update(dt)


def getArray(image):
    rawimage = image.get_image_data()
    colourformat = "I"
    pitch = rawimage.width * len(colourformat)
    return rawimage.get_data(colourformat, pitch)


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


def save_model(tup):
    kompos = tup[0].find(",")
    path = modelpath + f"type{tup[0][:kompos]}"
    if not exists(path):
        mkdir(path)
    tup[1].save(path + fs + f"model{tup[0][kompos+1:]}.h5")


def set_car():
    global car  # skipcq: PYL-W0603
    car = []
    if Human:
        car.append(
            Car(
                rotation=-90,
                scale=(game_window.height / 600 + game_window.width / 800) / 6,
                img=car_image,
                x=450,
                y=80,
            )
        )
        car[0].IFLocal.LevelLayer_width = LevelLayer.width
        car[0].IFLocal.LevelLayer_height = LevelLayer.height
        car[0].IFLocal.pixels = pixels
        car[0].NN = NNHuman()
        car[0].NN.Interface = car[0].IFLocal

    index = Human
    for i in range(50):
        for x in range(100):
            if f"{i},{x}" in models:
                car.append(
                    Car(
                        rotation=-90,
                        scale=(game_window.height / 600 + game_window.width / 800) / 6,
                        img=car_image,
                        x=450,
                        y=80,
                    )
                )

                car[index].nViewlines = 8
                car[index].IFLocal.modelindex = [i, x]
                car[index].IFLocal.model = models[f"{i},{x}"]
                car[index].IFLocal.LevelLayer_width = LevelLayer.width
                car[index].IFLocal.LevelLayer_height = LevelLayer.height
                car[index].IFLocal.pixels = pixels
                if not (i == 0 and Human) and isinstance(car[index].NN, NNev):
                    car[index].NN.set_model()
                index += 1


def empty_room():
    global generation, models, model_scores  # skipcq: PYL-W0603
    generation += 1
    tot_time = int(time.time()) - start_time
    print(f"gen {generation} completed, {tot_time} sec passed in total")
    top = ceil((nCars - Human) / 2)
    best = [
        (k, model_scores[k])
        for k in sorted(model_scores, key=model_scores.get, reverse=True)
    ]

    indexes = [0]
    app = indexes.append
    for n in range(1, len(best)):
        if best[n - 1] != best[n]:
            app(n)
    for x in range(len(indexes) - 1):
        copy = best[indexes[x] : indexes[x + 1]]
        shuffle(copy)
        best[indexes[x] : indexes[x + 1]] = copy

    best = best[:top]
    print("best: ", best)
    index = Human
    models2 = {}
    for i in range(50):
        for x in range(100):
            if f"{i},{x}" in models:
                if all(f"{i},{x}" != elem[0] for elem in best):
                    bestindex = index % top
                    best_model = models[best[bestindex][0]]
                    del models[f"{i},{x}"]

                    mutation_rate = 1 / ((25 + best[bestindex][1]) / 25)
                    nexti = int(best[bestindex][0][0])
                    if randint(0, int(nCars * (best[index % top][1] + 1) ** 1.5)) == 0:
                        print("Model mutated")
                        mutated_model = gen_mutant(
                            generate_random_NNev_model(best_model, mutation_rate),
                            mutation_rate,
                        )
                        print(f"nexti before: {nexti}")
                        nexti += 1
                        for p in range(50):
                            if not (
                                f"{p},0" in models
                                or f"{p},1" in models
                                or f"{p},0" in models2
                                or f"{p},1" in models2
                            ):
                                nexti = p
                                break
                        print(f"nexti after: {nexti}")
                    else:
                        mutated_model = gen_mutant(best_model, mutation_rate)
                    nextx = int(best[bestindex][0][2]) + 1
                    for p in range(100):
                        if not (f"{nexti},{p}" in models or f"{nexti},{p}" in models2):
                            nextx = p
                            break
                    models2[f"{nexti},{nextx}"] = mutated_model
                    index += 1
    models.update(models2)
    if generation % 10 == 0:
        if exists(modelpath):
            rmtree(modelpath, ignore_errors=True)
        while exists(modelpath):  # check if it exists
            pass
        mkdir(modelpath)
        with open(modelpath + "gen.log", "w") as f:
            f.write(f"[{generation}, {tot_time}]")
        for item in models.items():
            save_model((item[0], item[1]))
    model_scores = {}
    set_car()


def collision(index):
    global car  # skipcq: PYL-W0603
    if index == 0 and Human:
        del car[0]
        car.insert(
            0,
            Car(
                rotation=-90,
                scale=(game_window.height / 600 + game_window.width / 800) / 6,
                img=car_image,
                x=450,
                y=80,
            ),
        )
        car[0].nViewlines = 10 * (
            0 if Human == 1 and index == 0 else 1
        )  # (index+1-Human) * 4
        car[0].IFLocal.LevelLayer_width = LevelLayer.width
        car[0].IFLocal.LevelLayer_height = LevelLayer.height
        car[0].IFLocal.pixels = pixels
        car[0].NN = NNHuman()
        car[0].NN.Interface = car[0].IFLocal
    else:
        model_scores[
            f"{car[index].IFLocal.modelindex[0]},{car[index].IFLocal.modelindex[1]}"
        ] = car[index].score
        car[index] = None

    print("boem", end="\r")
    if all((elem is None) for elem in car):
        empty_room()


center_image(car_image)
game_window = pyglet.window.Window(Level.width, Level.height)

pixels = getArray(LevelLayer)
inp = input("use previous models? ")
while all(inp != ans for ans in ["yes", "no", "y", "n", "Yes", "No", "Y", "N"]):
    inp = input("use previous models? ")

if any(inp == ans for ans in ["no", "n", "No", "N"]):
    print("ok, no then")
    model_scores = {}
    for modtype in range(nModels):
        model = generate_random_NNev_model()
        print(modtype)
        for modnum in range(nCarsPerModel):
            models[f"{modtype},{modnum}"] = gen_mutant(gen_mutant(model, 1.0), 1.0)
else:
    with open(modelpath + "gen.log", "r") as logfile:
        data = literal_eval(logfile.read())
        generation = data[0]
        start_time = int(time.time()) - data[1]
    for modtype in range(50):
        if exists(f"{modelpath}type{modtype}"):
            for modnum in range(100):
                if exists(f"{modelpath}type{modtype}{fs}model{modnum}.h5"):
                    models[f"{modtype},{modnum}"] = load_model(
                        f"{modelpath}type{modtype}{fs}model{modnum}.h5"
                    )
    model_scores = {}
set_car()
update()


@game_window.event
def on_key_press(key, modifiers):
    global b, inf, showmodel, show_cars  # skipcq: PYL-W0603
    if key == pyglet.window.key.LEFT:
        car[0].NN.output[0] = 1
    elif key == pyglet.window.key.RIGHT:
        car[0].NN.output[1] = 1
    elif key == pyglet.window.key.UP:
        car[0].NN.output[2] = 1
    elif key == pyglet.window.key.DOWN:
        car[0].NN.output[3] = 1
    elif key == pyglet.window.key.B:
        b = not bool(b)
    elif key == pyglet.window.key.I:
        inf = not bool(inf)
    elif key == pyglet.window.key.S:
        showmodel = not bool(showmodel)
    elif key == pyglet.window.key.C:
        show_cars = not bool(show_cars)


@game_window.event
def on_key_release(key, modifiers):
    if key == pyglet.window.key.LEFT:
        car[0].NN.output[0] = 0
    elif key == pyglet.window.key.RIGHT:
        car[0].NN.output[1] = 0
    elif key == pyglet.window.key.UP:
        car[0].NN.output[2] = 0
    elif key == pyglet.window.key.DOWN:
        car[0].NN.output[3] = 0


@game_window.event
def on_draw():
    game_window.clear()
    if b:
        Level.blit(0, 0)
    else:
        LevelLayer.blit(0, 0)
    pyglet.text.Label(
        "FPS: {:4.2f}".format(pyglet.clock.get_fps()),
        font_name="Times New Roman",
        font_size=10,
        x=Level.width - 100,
        y=Level.height - 40,
    ).draw()
    for count, elem in enumerate(car):
        if elem:
            if show_cars:
                elem.draw()
                elem.draw_info(10, game_window.height - 20 * count - 20, not inf)
                if showmodel and car[0]:
                    car[0].show_model()
                pyglet.text.Label(
                    "{:d}, {:d}".format(
                        elem.IFLocal.modelindex[0], elem.IFLocal.modelindex[1]
                    ),
                    font_name="Times New Roman",
                    font_size=10,
                    x=elem.x,
                    y=elem.y,
                ).draw()
            if elem.check_collision(elem.hitbox) or (
                not (Human and count == 0) and elem.unusedframes > 100
            ):
                collision(count)


pyglet.clock.schedule_interval(update, 1 / 120.0)

if __name__ == "__main__":
    pyglet.app.run()
