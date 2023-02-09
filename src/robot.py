import constants
from shapely.geometry import LineString, Point
import math
import random
import numpy as np
from field import Field

class Robot(object):
	#TODO: add noise to move and sense
	#set noise by random.gauss function by using random.gauss(0.0, (insert noise var))
	def __init__(self, randomPos=False, specPos=None, specOrientation=0.0, waypoints=[], bag_mode=False):
		#init robot at a coordinate x, y
		#print("Initializing")
		self.bag_mode = bag_mode
		self.env_size = constants.ENVIRONMENT_SIZE
		self.lidar_inc = math.radians(0.25) # lidar 0.25 deg sample
		
		if specPos is not None:
			self.x = specPos[0]
			self.y = specPos[1]
		else:
			while True:
				self.x = (random.random()*randomPos * self.env_size)
				self.y = (random.random()*randomPos * self.env_size)

				in_wall = False if bag_mode else self.hit_wall()

				if not in_wall:
					break

		self.est_x = self.x
		self.est_y = self.y

		self.orientation = specOrientation
		#self.landmarks = Field.landmarks
		self.forward_noise = 0.0
		self.turn_noise = 0.0
		self.sense_noise = 0.0

		# for setting set trajectories
		self.waypoints = waypoints
		self.move_to_next = False

	def move_to_waypoint(self, max_speed=3):
		if len(self.waypoints) != 0:
			waypoint = self.waypoints[0]
			target_orientation = (np.arctan2(waypoint[1] - self.y, waypoint[0] - self.x)) % (2*math.pi)
			if not self.move_to_next:
				if abs(target_orientation-self.orientation) <= math.pi/5:
					self.orientation = target_orientation
					self.move_to_next = True
				else:
					self.orientation += np.sign(target_orientation - self.orientation) * math.pi/10
				return 0
			else:
				self.orientation = target_orientation

			dist = math.sqrt((self.x - waypoint[0]) ** 2 + (self.y - waypoint[1]) ** 2)
			if dist < max_speed:
				speed = dist
			else:
				speed = max_speed
			self.move(speed, 0)

			# check if passed waypoint
			if math.sqrt((self.x - waypoint[0]) ** 2 + (self.y - waypoint[1]) ** 2) < 1:
				self.waypoints.pop(0)
				self.move_to_next = False

				if len(self.waypoints) != 0:
					print('NEXT WAYPOINT: (%i, %i)' % self.waypoints[0])
		return speed

	def move(self, fwd, heading, ignore_walls=False):
		self.orientation += heading + random.gauss(0.0, self.turn_noise)
		self.orientation %= 2*math.pi
 
		x_fwd = math.cos(self.orientation)*fwd + random.gauss(0.0, self.forward_noise)
		y_fwd = math.sin(self.orientation)*fwd + random.gauss(0.0, self.forward_noise)

		old_x = self.x
		old_y = self.y

		self.x += x_fwd
		self.y += y_fwd
		self.x %= self.env_size
		self.y %= self.env_size
		
		if not ignore_walls and (self.hit_wall_on_move(old_x, old_y) or self.bag_mode):
			self.x -= x_fwd
			self.y -= y_fwd
			self.orientation += math.pi

		rob = Robot()
		rob.set(self.x, self.y, self.orientation)
		rob.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
		return rob

	def set(self, new_x, new_y, new_orientation):
		self.x = new_x
		self.y = new_y
		self.orientation = new_orientation

	def hit_wall_on_move(self, old_x, old_y):
		for wall in Field.walls:
			wall_a = wall[0]
			wall_b = wall[1]

			if self.lineLine(old_x, old_y, self.x, self.y, wall_a[0], wall_a[1], wall_b[0], wall_b[1]):
				return True

		return False

	def hit_wall(self):
		for wall in Field.walls:
			wall_a = wall[0]
			wall_b = wall[1]

			r = (self.x, self.y)

			ab = [wall_a[0]-wall_b[0], wall_a[1]-wall_b[1]]
			ar = [wall_a[0]-r[0], wall_a[1]-r[1]]

			if abs(np.cross(ab, ar)) == 0:
				return True

		return False

	def getPos(self):
		#TODO: Return the position of the robot
		return [self.x, self.y]
	 
	def lineLine(self, x1, y1, x2, y2, x3, y3, x4, y4):
		uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
		uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))

		if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
			return True
		return False

	def isColliding(self, x1, y1, x2, y2, rx1, ry1, rx2, ry2):
		left = self.lineLine(x1, y1, x2, y2, rx1, ry2, rx1, ry1)
		right = self.lineLine(x1, y1, x2, y2, rx2, ry2, rx2, ry1)
		top = self.lineLine(x1, y1, x2, y2 , rx2, ry2, rx1, ry2)
		bottom = self.lineLine(x1, y1, x2, y2, rx2, ry1, rx1, ry1)
	
		if left or right or top or bottom:
			return True
		return False

	def lidar_match(self, theta, lidar_data):
		for datapoint in lidar_data:
			# get range where datapoint can be used
			if theta >= datapoint[1]-self.lidar_inc and theta <= datapoint[1]+self.lidar_inc:
				return datapoint

	def canSense(self, landmark):
		return True

	def sense(self, occupancy_grid, lidar_data):
		dists = []
		for landmark in occupancy_grid.landmarks:
			if self.canSense(landmark): 
				theta_to_landmark = np.arctan2(landmark[1] - self.est_y, landmark[0] - self.est_x)
				# wrap +pi to -pi
				if theta_to_landmark > np.pi:
					theta_to_landmark -= 2*np.pi

				if theta_to_landmark < -np.pi:
					theta_to_landmark += 2*np.pi

				dist = self.lidar_match(theta_to_landmark, lidar_data)
				if dist is None:
					dists.append(-1)
					continue
				dist = dist[0]
				dist += random.gauss(0.0, self.sense_noise)
				dists.append(dist)
			else:
				dists.append(-1)
		return dists

	def lidar_sense(self):
		# 180 degree cone from front of robot (sensor is 270 but 180 can only be used)
		cur_rad = self.orientation - math.pi / 2
		end_rad = self.orientation + math.pi / 2
		
		lidar_data = None
		intersect_points = []

		while cur_rad < end_rad:
			adj_rad = cur_rad % (math.pi*2)
			# assume lidar can reach entire field
			r_line = np.array([[self.x, self.y], [self.x+(math.cos(adj_rad)*constants.ENVIRONMENT_SIZE*2), self.y+(math.sin(adj_rad)*constants.ENVIRONMENT_SIZE*2)]])
			
			# to optimize, create a set of walls that are possible to intersect to reduce space complexity
			wall_intersect = None
			min_dist = constants.ENVIRONMENT_SIZE*2
			for wall in Field.walls:
				intersect = self.line_intersection(r_line, wall) # TODO: REALLY SLOW find better way to improve

				if intersect is not None:
					distance = math.sqrt((self.x-intersect[0])**2 + (self.y-intersect[1])**2)
					if min_dist > distance:	
						min_dist = distance
						wall_intersect = intersect
			
			if wall_intersect is None:
				raise Exception('In this environment, not possible to not have intersection')

			angle = np.arctan2(wall_intersect[1] - self.y, wall_intersect[0] - self.x)
			
			# wrap +pi to -pi
			if angle > np.pi:
				angle -= 2*np.pi

			if angle < -np.pi:
				angle += 2*np.pi

			if lidar_data is not None:
				lidar_data = np.vstack([lidar_data, np.array([min_dist, angle])])
			else:
				lidar_data = np.array([[min_dist, angle]])

			intersect_points.append(wall_intersect)

			cur_rad += self.lidar_inc

		return lidar_data, intersect_points

	def seg_intersect(self, line1, line2) :
		def perp( a ):
			b = np.empty_like(a)
			b[0] = -a[1]
			b[1] = a[0]
			return b
		
		a1, a2 = line1[0], line1[1]
		b1, b2 = line2[0], line2[1]

		da = a2-a1
		db = b2-b1
		dp = a1-b1
		dap = perp(da)
		denom = np.dot( dap, db)
		num = np.dot( dap, dp )
		return (num / denom.astype(float))*db + b1

	def line_intersection(self, l1, l2):
		line1 = LineString([l1[0], l1[1]])
		line2 = LineString([l2[0], l2[1]])

		int_pt = line1.intersection(line2)

		# check if point found (no point found would not define x and y)
		try:
			return int_pt.x, int_pt.y
		except:
			return None

	# def line_intersection(self, l1, l2):
	# 	def line_coefs(line):
	# 		p1 = line[0]
	# 		p2 = line[1]
	# 		A = p1[1] -p2[1]
	# 		B = p2[0] - p1[0]
	# 		C = p1[0]*p2[1] - p2[0]*p1[1]
	# 		return A, B , -C
		
	# 	l1_coefs = line_coefs(l1)
	# 	l2_coefs = line_coefs(l2)

	# 	# Cramer's Rule (x = Dx / D, y = Dy/D)
	# 	D = l1_coefs[0] * l2_coefs[1] - l1_coefs[1] * l2_coefs[0]
	# 	Dx = l1_coefs[2] * l2_coefs[1] - l1_coefs[1] * l2_coefs[2]
	# 	Dy = l1_coefs[0] * l2_coefs[2] - l1_coefs[2] * l2_coefs[0]

	# 	if D != 0:
	# 		return Dx/D, Dy/D
	# 	return None

	def Gaussian(self, mu, sigma, x):
		return math.exp(-((mu-x) ** 2) / (sigma ** 2)/2.0) / math.sqrt(2.0 * math.pi * (sigma ** 2))

	def measurement_prob(self, measurement, occupancy_grid):
		prob = 1.0
		for i in range(len(occupancy_grid.landmarks)):
			if measurement[i] == -1:
				continue
			landmark = occupancy_grid.landmarks[i]
			dist = math.sqrt((self.x - landmark[0])**2 + (self.y - landmark[1]) ** 2)
			prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
		return prob

	def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
		#TODO: set noise to three vars called forward_noise, turn_noise, sense_noise as floats
		self.forward_noise = new_f_noise
		self.turn_noise = new_t_noise
		self.sense_noise = new_s_noise

	def get_estimated_state(self):
		return np.array([self.est_x, self.est_y, self.orientation])
	
	def set_estimated_state(self, x, y):
		self.est_x = x
		self.est_y = y

	def eval(self, r, p):
		sum = 0.0
		for i in range(len(p)):
			dx = (p[i].x - r.x + (self.env_size/2.0)) % self.env_size - (self.env_size/2.0)
			dy = (p[i].y - r.y + (self.env_size/2.0)) % self.env_size - (self.env_size/2.0)
			err = math.sqrt(dx * dx + dy * dy)
			sum += err
		return sum / float(len(p))

	def get_state(self):
		return np.array([self.x, self.y, self.orientation])

	def set_state(self, x, y, orientation):
		self.x = x
		self.y = y
		self.orientation = orientation

	def __repr__(self):
		return("X: %f Y: %f Heading: %f" % (self.x, self.y, self.orientation))
