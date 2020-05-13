from math import sqrt, pi, atan2, radians
import numpy as np
from numpy import array as nparray
from numpy import around
from numpy.random import *
seed(777)
from keras.models import Sequential, clone_model, model_from_json
from keras.layers import Dense, Activation
from classes.Interface import Interface
import sys
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

def AngleBetween(a, b):
	p2 = 2 * pi

	angle = 0

	k = b - a
	l = b - p2 - a
	m = b + p2 - a

	ak = abs(k)
	al = abs(l)
	am = abs(m)

	mini = min(ak,al)
	mini = min(mini, am)

	angle = m
	if mini == ak: angle = k
	if mini == al: angle = l

	return angle

class NN():
	def __init__(self):
		self.Interface: Interface()

class NNLongest(NN):
	def __init__(self):
		super().__init__()

	def find_longest(self):
		longest = [0, 0]
		xc, yc = self.Interface.xc, self.Interface.yc
		i = 0
		viewline = self.Interface.viewline
		for line in viewline:
			if line != []:
				length = sqrt((line[0] - xc)**2 + (line[1] - yc)**2)
				if length > longest[0]:
					longest = [length, i]
			i += 1
		return longest[1]

	def find_shortest(self):
		shortest = [10000, 0]
		i = 0
		xc, yc = self.Interface.xc, self.Interface.yc
		viewline = self.Interface.viewline
		for line in viewline:
			if line != []:
				length = sqrt((line[0] - xc)**2 + (line[1] - yc)**2)
				if length < shortest[0]:
					shortest = [length, i]
			i += 1
		return shortest

	def run(self):
		xc, yc = self.Interface.xc, self.Interface.yc
		if xc != 0 and len(self.Interface.viewline[0]) != 0:
			longest = self.find_longest()
			viewline = self.Interface.viewline
			xl, yl = self.Interface.viewline[longest][0], self.Interface.viewline[longest][1]
			rotations = self.Interface.rotations

			angle = atan2((yl - yc), (xl - xc))
			diffl = AngleBetween(radians(90-rotations), angle)

			shortest = self.find_shortest()

			xs, ys = self.Interface.viewline[shortest[1]][0], self.Interface.viewline[shortest[1]][1]

			angles = atan2((ys - yc), (xs - xc))
			diffs = AngleBetween(radians(90-rotations), angles)
			if shortest[0] > self.Interface.carWidth:
				if diffl > 0: self.Interface.input = [1,0,1,0]
				elif diffl < 0: self.Interface.input = [0,1,1,0]
				else: self.Interface.input = [0,0,1,0]
			else:
				if diffs < 0: self.Interface.input = [1,0,1,0]
				elif diffs > 0: self.Interface.input = [0,1,1,0]
				else: self.Interface.input = [0,0,1,0]


class NNHuman(NN):
	def __init__(self):
		super().__init__()
		self.output = [0,0,0,0]

	def run(self):
		self.Interface.input = self.output

def generate_random_NNev_model(pre_model = [] , mutation_rate = 0):
	activations = ['elu','softmax','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','exponential','linear']
	if pre_model == []:
		layers = randint(low=0, high=20) + 2
		nodes = [11]
		nodes.extend([randint(low=1, high=200) for p in range(layers-2)])
		nodes.append(4)
		activation = [choice(activations) for i in range(layers-1)]
		activation.append('sigmoid')
	else:
		premodel = [len(pre_model.layers[1::2]), [l.output_shape[1] for l in pre_model.layers[1::2]], [l.get_config()['activation'] for l in pre_model.layers[1::2]]]
		if binomial(1, mutation_rate):
			layers = premodel[0] + randint(low=(2-premodel[0])//(8/mutation_rate), high=(20-premodel[0])//(8/mutation_rate))
		else:
			layers = premodel[0]

		nodes = premodel[1]
		if len(nodes) < layers:
			nodes.pop()
			nodes.extend([randint(low=1, high=200) for p in range(layers-len(nodes))])
			nodes.append(4)
		elif len(nodes) > layers:
			nodes = nodes[:-(len(nodes)-layers+1)]
			nodes.append(4)
		if binomial(1, mutation_rate):
			for x in range(randint(low=0, high=1+len(nodes)//(8/mutation_rate))):
				nodes[randint(0,len(nodes)-1)] = randint(min(nodes),max(nodes)-1)

		activation = premodel[2]
		if len(activation) < layers:
			activation.pop()
			activation.extend([choice(activations) for p in range(layers-len(activation))])
			activation.append('sigmoid')
		elif len(activation) > layers:
			activation = activation[:-(len(activation)-layers+1)]
			activation.append('sigmoid')
		for x in range(randint(low=0, high=len(activation)//(8/mutation_rate)+1)):
			activation[randint(0,len(nodes)-2)] = choice(activations)

	#print(layers,'\n')
	#print(nodes,'\n')
	#print(activation,'\n')
	model = Sequential()
	add = model.add
	add(Dense(nodes[0], input_dim=1))
	add(Activation(activation[0]))
	for l in range(1, layers):
		add(Dense(nodes[l]))
		add(Activation(activation[l]))

	#print(len(model.layers[1::2]),'\n')
	#print([l.output_shape[1] for l in model.layers[1::2]],'\n')
	#print([l.get_config()['activation'] for l in model.layers[1::2]],'\n')

	return model

def modify_weights(weight_val, mutation_rate):
	#print('weight_val before: ', weight_val)
	if weight_val.ndim == 2:
		for x in range(len(weight_val)):
			try:
				rand_int = randint(0, len(weight_val[x])-1)
				weight_val[x][rand_int] = normal(weight_val[x][rand_int], abs(weight_val[x][rand_int] * mutation_rate) if weight_val[x][rand_int] != 0 else mutation_rate)
			except ValueError: weight_val[x][0] = normal(weight_val[x][0], abs(weight_val[x][0] * mutation_rate) if weight_val[x][0] != 0 else 1)
	else:
		try:
			rand_int = randint(0, len(weight_val)-1)
			weight_val[rand_int] = normal(weight_val[rand_int], abs(weight_val[rand_int] * mutation_rate) if weight_val[rand_int] != 0 else mutation_rate)
		except ValueError: weight_val[0] = normal(weight_val[0], abs(weight_val[0] * mutation_rate) if weight_val[0] != 0 else mutation_rate)
	#print('weight_val after: ', weight_val)
	return weight_val


def gen_mutant(parent_model, mutation_rate):
	new_weights = parent_model.get_weights()
	#print('old new_weights[0]: ', new_weights[0])
	for weight_array in new_weights:
		num_weights = weight_array.size if weight_array.ndim == 1 else weight_array[0].size
		num_weights_modified = binomial(num_weights, mutation_rate)
		for i in range(num_weights_modified):
			weight_array = modify_weights(weight_array, mutation_rate)
	mutant = clone_model(parent_model)
	#print('new new_weights[0]: ', new_weights[0])
	mutant.set_weights(new_weights)
	return mutant

class NNev(NN):
	def __init__(self):
		super().__init__()
		self.model: sequential()

	def set_model(self):
		print('mmodel set')
		model = self.Interface.model
		self.model = model
		#print(self.model)

	def run(self):
		try:
			vl = self.Interface.viewline
			iface = self.Interface
			input_data = [iface.speed]
			app = input_data.append
			for i in range(10):
				app(sqrt((iface.xc - vl[i][0])**2 + (iface.yc - vl[i][1])**2))


			#print('input_data: ',input_data)
			output = self.model.predict(nparray(input_data))[0]
			#print('output: ',output)
			output = around(output,  decimals=0)
			#print('rounded output: ',output)

			self.Interface.input = [output[0],output[1],output[2],output[3]]
		except IndexError:
			pass
