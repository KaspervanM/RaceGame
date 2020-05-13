import pyglet
from math import sqrt, cos, sin, acos, pi, degrees, radians
from classes.Interface import Interface
from classes.NN import *

class Car(pyglet.sprite.Sprite):
	def __init__(self, rotation = 0, scale = 1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.scale	= scale
		self.IFLocal 	= Interface(0, [0,0,0,0], [0,0,0,0], [], 0, 0, 0, 0, [], 0, self.width, 0, [-1, -1], 0)
		self.NN 	= NNev()
		self.NN.Interface = self.IFLocal
		#self.NN.carWidth = self.width
		self.rotation	= rotation  # degrees
		self.velocity 	= (0,0)
		self.steer 	= 0.45*pi  # radians 0.5
		self.speedup 	= 1.5  # 1.5
		self.maxSpeed	= 80  # 200
		self.acc	= 0
		self.hitbox	= [0 for i in range(12)]  # [left bottom x, y, right bottom x, y, right top x, y, left top x, y]
		self.nViewlines = 10
		self.viewline	= [[] for i in range(self.nViewlines)]  # front, ffleft, fleft, ffright, fright, bottom-, left, right, bleft-, bright-
		self.score	= 0
		self.unusedframes = 0

	def setNNIF(self):
		self.NN.Interface = self.IFLocal

	def calc_score(self):
		intensity = self.IFLocal.pixels[int(self.x) + int(self.y) * self.IFLocal.LevelLayer_width]
		self.score = 255-intensity

	def check_collision(self, coords):
		pixels = self.IFLocal.pixels
		LevelLayer_width = self.IFLocal.LevelLayer_width
		for x in range(0, int(len(coords)) - 1, 2):
			if coords[x] + coords[x+1] * LevelLayer_width < len(pixels):
				if str(pixels[coords[x] + coords[x+1] * LevelLayer_width]) == '0':
					return True
		return False

	def calc_hitbox(self):
		height = self.height
		width = self.width
		rotation = self.rotation
		xc, yc = self.x, self.y
		xmt, ymt = int(xc-(height/2)*sin(2*pi-radians(rotation))), int(yc+(height/2)*cos(2*pi-radians(rotation)))
		xmb, ymb = int(xc-(height/2)*sin(pi-radians(rotation))), int(yc+(height/2)*cos(pi-radians(rotation)))
		xlt, ylt = int(xmt+(width/height)*(yc-ymt)), int(ymt+(width/height)*(xmt-xc))
		xrt, yrt = int(xmt+(width/height)*(ymt-yc)), int(ymt+(width/height)*(xc-xmt))
		xrb, yrb = int(xmb+(width/height)*(yc-ymb)), int(ymb+(width/height)*(xmb-xc))
		xlb, ylb = int(xmb+(width/height)*(ymb-yc)), int(ymb+(width/height)*(xc-xmb))
		self.hitbox = [xmt, ymt, xmb, ymb, xlb, ylb, xrb, yrb, xrt, yrt, xlt, ylt]

	def calc_viewlines(self):
		xc, yc   = self.x, self.y
		viewdist = self.IFLocal.LevelLayer_width+self.IFLocal.LevelLayer_height
		nViewlines = self.nViewlines
		rotation = self.rotation
		viewline = [[] for i in range(nViewlines)]
		vlck = [[] for i in range(nViewlines)]
		nvlck = [True for i in range(nViewlines)]
		anglefactors = [abs(i)/(i+0.000001)*(i/(nViewlines-1))**2*0.65 for i in range(-nViewlines+1, nViewlines, 2)]
		check_collision =  self.check_collision
		for d in range(0, viewdist, 4):
			for i in range(len(viewline)):
				if nvlck[i]:
					x, y = int(xc-d*sin(anglefactors[i]*pi-radians(rotation))), int(yc+d*cos(anglefactors[i]*pi-radians(rotation)))
					if d == viewdist-1 or check_collision([x, y]):
						viewline[i] = [x, y]
						nvlck[i] = False
				else: pass
		self.viewline = viewline

	def draw_info(self, x , y, textonly):
		velocity = self.velocity
		acc=self.acc
		rotation=self.rotation
		key=self.IFLocal.key
		X, Y = self.x, self.y
		# labels
		pyglet.text.Label('{num:d}, {num2:d}\tscore: {score:d}\tvector:({v1:4.2f},{v2:4.2f})\nspeed:{speed:4.2f}\nacc={acc:4.2f}\trot:{rot:4.2f}\nLRUP: {l:d}{r:d}{u:d}{d:d}\n(x,y):({X:4.2f},{Y:4.2f})'.format(
		 num=self.IFLocal.modelindex[0],num2=self.IFLocal.modelindex[1],v1=velocity[0], v2=velocity[1], speed=sqrt(velocity[0]**2 + velocity[1]**2), acc=acc, rot=rotation, l=key[0], r=key[1], u=key[2], d=key[3], X=X, Y=Y, score=self.score),
		 font_name='Times New Roman', font_size=10, x=x, y=y).draw()

		if not textonly:
			hitbox = self.hitbox
			# hitbox
			pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', hitbox[4:8]), ('c3B', (0, 255, 0, 0, 255, 0)))
			pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', hitbox[4:6]+hitbox[10:]), ('c3B', (0, 255, 0, 0, 255, 0)))
			pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', hitbox[6:10]), ('c3B', (0, 255, 0, 0, 255, 0)))
			pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', hitbox[8:]), ('c3B', (0, 255, 0, 0, 255, 0)))

			# vector
			xc, yc = X + velocity[0], Y + velocity[1]
			xmt, ymt = int(xc-(3)*sin(2*pi-radians(rotation))), int(yc+(3)*cos(2*pi-radians(rotation)))
			xmb, ymb = int(xc-(3)*sin(pi-radians(rotation))), int(yc+(3)*cos(pi-radians(rotation)))
			xrb, yrb = int(xmb+(1)*(yc-ymb)), int(ymb+(1)*(xmb-xc))
			xlb, ylb = int(xmb+(1)*(ymb-yc)), int(ymb+(1)*(xc-xmb))

			pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', [X, Y, X + velocity[0], Y + velocity[1]]))
			pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES, ('v2f', [xlb, ylb, xrb, yrb, xmt, ymt]))

			# vision
			viewline = self.viewline
			draw = pyglet.graphics.draw
			GL_LINES = pyglet.gl.GL_LINES
			for line in viewline:
				if line != []:
					if line[0] != []:
						draw(2, GL_LINES, ('v2f',[X, Y, line[0], line[1]]))

	def update(self, dt):
		if not isinstance(self.NN, NNHuman) and not isinstance(self.NN, NNLongest):
			try:
				if not self.NN.model:
					print('nomodel')
					return
			except AttributeError:
				print(AttributeError)
				return
		try: self.NN.run()
		except AttributeError: print('Model not run')
		self.IFLocal.set_code()
		velocity = self.velocity
		rotation=self.rotation
		key=self.IFLocal.key
		oX, oY = self.x, self.y
		X, Y = self.x, self.y
		speedup = self.speedup
		acc = (key[2]-key[3]) * speedup
		if key[2] == -1 and sqrt(velocity[0]**2 + velocity[1]**2) <= speedup :
			key[2] = 0
			velocity = (0,0)
		if key[3] == -2 and sqrt(velocity[0]**2 + velocity[1]**2) <= speedup :
			key[3] = 0
			velocity = (0,0)

		angle = self.steer * dt

		if key[0] == 1:
			velocity = (velocity[0] * cos(angle) - velocity[1] * sin(angle), velocity[0] * sin(angle) + velocity[1] * cos(angle))
		if key[1] == 1:
			velocity = (velocity[0] * cos(2*pi-angle) - velocity[1] * sin(2*pi-angle), velocity[0] * sin(2*pi-angle) + velocity[1] * cos(2*pi-angle))

		if sqrt(velocity[0]**2 + velocity[1]**2) > self.maxSpeed:
			if key[2] == 1: acc = 0

		c = 0
		if velocity[0] != 0 and velocity[1] != 0:
			a, b, d = velocity[0], velocity[1], acc
			#print(d)
			if (d**2+2*d*sqrt(a**2+b**2)+a**2+b**2) > 0:
				c = sqrt((d**2+2*d*sqrt(a**2+b**2)+a**2+b**2)/(a**2+b**2))
				velocity = (velocity[0] * c, velocity[1] * c)
		elif key[2] == 1:
			velocity = (sin(radians(rotation)), cos(radians(rotation)))

		X += velocity[0] * dt
		Y += velocity[1] * dt
		if velocity[0] != 0 or velocity[1] != 0:
			rotation = degrees(acos(velocity[1]/sqrt(velocity[0]**2 + velocity[1]**2)))
			if velocity[0] < 0: rotation = -rotation

		self.IFLocal.rotations = rotation
		self.IFLocal.speed = sqrt(velocity[0]**2 + velocity[1]**2)
		self.IFLocal.xc, self.IFLocal.yc = X, Y
		self.calc_viewlines()
		#print(self.viewline)
		#print([row[0] for row in self.viewline[:]])
		self.IFLocal.viewline = self.viewline
		self.calc_hitbox()

		self.calc_score()
		self.IFLocal.score = self.score

		self.rotation = rotation
		self.x = X
		self.y = Y
		self.velocity = velocity
		self.acc = acc
		if sqrt((oX - X)**2 + (oY - Y)**2) <= 0.8: self.unusedframes += 1
