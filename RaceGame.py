import pyglet
from keras.models import load_model
from os import mkdir
from os.path import exists
from shutil import rmtree
from collections import OrderedDict
from classes.Interface import Interface
from classes.Cars import Car
from classes.NN import *
import sys
import time
from ast import literal_eval
seed(777)
def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

fs = '\\' if get_platform() == 'Windows' else '/'
currdir = sys.path[0]
car_image = pyglet.image.load(currdir+fs+'resources'+fs+'car.png')
Level = pyglet.image.load(currdir+fs+'resources'+fs+'Level.png').get_texture()
LevelLayer = pyglet.image.load(currdir+fs+'resources'+fs+'LevelLayerWscore2.png').get_texture()
global b, inf, car
b = True
inf = False

Human = 0

nModels = 6 * (1-Human)
nCarsPerModel = 3 * (1-Human)
nCars = nModels * nCarsPerModel + Human
car = []

global generation, save_yet, models, model_scores
generation = 0
start_time = int(time.time())
save_yet = 0
models = {}
model_scores = []

def update(dt = 0):
	dt = 0.03
	for i in range(nCars):
		if car[i]:
			car[i].update(dt)

def getArray(image):
	rawimage = image.get_image_data()
	format = 'I'
	pitch = rawimage.width * len(format)
	return rawimage.get_data(format, pitch)

def center_image(image):
	"""Sets an image's anchor point to its center"""
	image.anchor_x = image.width // 2
	image.anchor_y = image.height // 2

def save_model(tup):
	path = currdir+fs+'resources'+fs+'models'+fs+f'type{tup[0][0]}'
	if not exists(path): mkdir(path)
	tup[1].save(path+fs+f'model{tup[0][2:]}.h5')

def set_car():
	global car
	car = []
	if Human:
		car.append(Car(rotation = -90, scale=(game_window.height / 600 + game_window.width / 800) / 6, img=car_image, x=450, y=80))
		car[0].IFLocal.LevelLayer_width = LevelLayer.width
		car[0].IFLocal.LevelLayer_height = LevelLayer.height
		car[0].IFLocal.pixels = pixels
		car[0].NN = NNHuman()
		car[0].NN.Interface = car[0].IFLocal

	index = Human
	for i in range(50):
			for x in range(100):
				if f"{i},{x}" in models:
					car.append(Car(rotation = -90, scale=(game_window.height / 600 + game_window.width / 800) / 6, img=car_image, x=450, y=80))

					car[index].nViewlines = 10 * (0 if Human and i == 0 else 1)  # (i+1-Human) * 4
					car[index].IFLocal.modelindex = [i, x]
					car[index].IFLocal.model = models[f"{i},{x}"]
					car[index].IFLocal.LevelLayer_width = LevelLayer.width
					car[index].IFLocal.LevelLayer_height = LevelLayer.height
					car[index].IFLocal.pixels = pixels
					if not (i == 0 and Human) and isinstance(car[index].NN, NNev): car[index].NN.set_model()
					index += 1

def empty_room():
	global generation, save_yet, models, model_scores, nModels
	generation += 1
	save_yet += 1
	tot_time = int(time.time()) - start_time
	print(f'gen {generation} completed, {tot_time} sec passed in total')
	top = (nCars-Human)//2
	best = [(k, model_scores[k]) for k in sorted(model_scores, key=model_scores.get, reverse=True)]
	most = (len(best)*3)//4
	if all(s[1] == best[0][1] for s in best[:most]):
		best = best[:most]
		shuffle(best)
	best = best[:top]
	print('best: ', best)
	index = Human
	models2 = {}
	for i in range(50):
			for x in range(100):
				if f"{i},{x}" in models:
					if all(f"{i},{x}" != elem[0] for elem in best):
						best_model = models[f"{i},{x}"]
						del models[f"{i},{x}"]

						bestindex = index % top
						I = int(best[bestindex][0][0])
						if randint(0, nCars*best[index % top][1]**1.5) == 0:
							print('Model mutated')
							mutated_model = generate_random_NNev_model(best_model, 1/((10+best[bestindex][1])/10))
							print(f"I before: {I}")
							I += 1
							for p in range(50):
								if not (f"{p},0" in models or f"{p},1" in models or f"{p},0" in models2 or f"{p},1" in models2):
									I = p
									break
							print(f"I after: {I}")
						else:
							mutated_model = gen_mutant(best_model, 1/((10+best[bestindex][1])/10))
						nextx = int(best[bestindex][0][2])+1
						for p in range(100):
							if not (f"{I},{p}" in models or f"{I},{p}" in models2):
								nextx = p
								break
						models2[f"{I},{nextx}"] = mutated_model
						index += 1
	models.update(models2)
	if save_yet == 10:
		path = currdir+fs+'resources'+fs+'models'
		if exists(path): rmtree(path, ignore_errors=True)
		while exists(path): # check if it exists
			pass
		mkdir(path)
		with open(currdir+fs+'resources'+fs+'models'+fs+'gen.log', 'w') as f:
			f.write(f'[{generation}, {tot_time}]')
		for key in models:
			save_model((key, models[key]))
		save_yet = 0
	model_scores = {}
	set_car()

def collision(index):
	global car
	if index == 0 and Human:
		del car[0]
		car.insert(0, Car(rotation = -90, scale=(game_window.height / 600 + game_window.width / 800) / 6, img=car_image, x=450, y=80))
		car[0].nViewlines = 10 * (0 if Human == 1 and index == 0 else 1)  # (index+1-Human) * 4
		car[0].IFLocal.LevelLayer_width = LevelLayer.width
		car[0].IFLocal.LevelLayer_height = LevelLayer.height
		car[0].IFLocal.pixels = pixels
		car[0].NN = NNHuman()
		car[0].NN.Interface = car[0].IFLocal
	else:
		model_scores[f"{car[index].IFLocal.modelindex[0]},{car[index].IFLocal.modelindex[1]}"] = car[index].score
		car[index] = None

	print("boem", end='\r')
	if all((elem == None) for elem in car):
		empty_room()

center_image(car_image)
game_window = pyglet.window.Window(Level.width, Level.height)

pixels = getArray(LevelLayer)
inp = input("use previous models? ")
while all(inp != ans for ans in ['yes', 'no', 'y', 'n', 'Yes', 'No', 'Y', 'N']):
	inp = input("use previous models? ")

if any(inp == ans for ans in ['no', 'n', 'No', 'N']):
	print('ok, no then')
	model_scores = {}
	for i in range(nModels):
		model = generate_random_NNev_model()
		print(i)
		for x in range(nCarsPerModel):
			models[f"{i},{x}"] = gen_mutant(gen_mutant(model, 1.), 1.)
else:
	path = currdir+fs+'resources'+fs+'models'+fs
	with open(path+'gen.log', 'r') as f:
		data = literal_eval(f.read())
		generation = data[0]
		start_time = int(time.time())-data[1]
	for i in range(nModels):
		if exists(f"{path}type{i}"):
			for x in range(100):
				if exists(f"{path}type{i}{fs}model{x}.h5"):
					models[f"{i},{x}"] = load_model(f"{path}type{i}{fs}model{x}.h5")
	model_scores = {}
set_car()
update()

@game_window.event
def on_key_press(key, modifiers):
	global b, inf
	if(key == pyglet.window.key.LEFT):
		car[0].NN.output[0] = 1
	elif(key == pyglet.window.key.RIGHT):
		car[0].NN.output[1] = 1
	elif(key == pyglet.window.key.UP):
		car[0].NN.output[2] = 1
	elif(key == pyglet.window.key.DOWN):
		car[0].NN.output[3] = 1
	elif(key == pyglet.window.key.B):
		b = False if b else True
	elif(key == pyglet.window.key.I):
		inf = False if inf else True

@game_window.event
def on_key_release(key, modifiers):
	if(key == pyglet.window.key.LEFT):
		car[0].NN.output[0] = 0
	elif(key == pyglet.window.key.RIGHT):
		car[0].NN.output[1] = 0
	elif(key == pyglet.window.key.UP):
		car[0].NN.output[2] = 0
	elif(key == pyglet.window.key.DOWN):
		car[0].NN.output[3] = 0


@game_window.event
def on_draw():
	game_window.clear()
	if b: Level.blit(0,0)
	else: LevelLayer.blit(0,0)
	pyglet.text.Label('FPS: {:4.2f}'.format(pyglet.clock.get_fps()),
	 font_name='Times New Roman', font_size=10, x=Level.width - 100, y=Level.height-40).draw()
	for i in range(len(car)):
		if car[i]:
			car[i].draw()
			car[i].draw_info(10,game_window.height - 20*i - 20, not inf)
			pyglet.text.Label('{:d}, {:d}'.format(car[i].IFLocal.modelindex[0], car[i].IFLocal.modelindex[1]),
			 font_name='Times New Roman', font_size=10, x=car[i].x, y=car[i].y).draw()
			if (car[i].check_collision(car[i].hitbox) or (not (Human and i == 0) and car[i].unusedframes > 100)):
				collision(i)
pyglet.clock.schedule_interval(update, 1/120.0)

if __name__ == '__main__':
	pyglet.app.run()
