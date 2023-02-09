import numpy as np

class Field(object):
	@classmethod
	def init(cls):
		#sets the field as a square with length field_size
		cls.size = 40

		#set example landmark
		cls.landmarks = [[0, 0], [50, 39], [60, 39], [65, 0], [0, 40], [99, 99], [0,99], [99,0], [40, 61], [50, 61], [50, 99]] 

		cls.walls = np.array([
			[[5,5], [5,95]],
			[[5,50], [50,50]],
			[[50,50], [70,50]],
			[[80,50], [95,50]],
			[[50,50], [50,70]],
			[[50,80], [50,95]],
			[[95,50], [50,5]],
			[[5,5], [50,5]],
			[[5,95], [95,95]],
			[[95,50], [95,95]]
		])