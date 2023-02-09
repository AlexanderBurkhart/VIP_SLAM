from pdb import lasti2lineno
import constants
from field import Field
from occupancy_grid import Occupancy_Grid
from particle_filter import ParticleFilter
from robot import Robot
import random
import math
import cv2
import numpy as np
import time

import matplotlib.pyplot as plt

Field.init()
def scale_point(point):
	return tuple([int(pos*SCALING)+BORDER for pos in point])

occupancy_grid = Occupancy_Grid(constants.ENVIRONMENT_SIZE, constants.ENVIRONMENT_SIZE, 1.0)

starting_pos = (10, 10)
waypoints = [
	(30, 40),
	(75, 40),
	(75,75),
	(20, 75),
	(20, 60),
	(20, 75),
	(75, 75),
	(75, 40),
	(30, 40),
	(10, 10)
]

robot = Robot(specPos=starting_pos, waypoints=waypoints)
print(robot.getPos())

robot.set_noise(0.05, 0.05, 0.05)

ground_truth = []
best_particle_pos = []
all_particles_pos = []
all_lidar_data = []
all_visual_intersect_data = []
log_prob_maps = []
landmarks = []

T = 100
N = 1000

p_filter = ParticleFilter(robot, N)

for t in range(T):
	# p = []
	# for i in range(N):
	# 	x = Robot(specPos=robot.get_estimated_state())
	# 	x.set_noise(0.05, 0.05, 5.0)
	# 	p.append(x)
	if len(robot.waypoints) == 0:
		break
	speed = robot.move_to_waypoint(1)

	lidar_data, visual_intersect_data = robot.lidar_sense()
	all_visual_intersect_data.append(visual_intersect_data)
	all_lidar_data.append(lidar_data)

	occupancy_grid.update_map(robot.get_estimated_state(), lidar_data)
	landmarks.append(occupancy_grid.landmarks) 

	dists = robot.sense(occupancy_grid, lidar_data)

	best_p, p = p_filter.step(occupancy_grid, dists, robot, speed)
	
	log_prob_maps.append(occupancy_grid.log_prob_map.copy())

	ground_truth.append(robot.get_state())
	best_particle_pos.append((best_p.getPos()[0], best_p.getPos()[1]))
	all_particles_pos.append(p)
	robot.set_estimated_state(best_p.getPos()[0], best_p.getPos()[1])
	print("%i: Actual Pos:[X: %f Y: %f] Particle Pos:[X: %f Y: %f] Orientation: %f" % (t, robot.x, robot.y, best_p.getPos()[0], best_p.getPos()[1], robot.orientation)) 

SIZE = 800
BORDER = 10
SCALING = SIZE / constants.ENVIRONMENT_SIZE
env_img = None
grid_img = None
idx = 0
while True:
	if cv2.waitKey(33) == ord('q'):
		break

	if idx >= len(ground_truth):
		time.sleep(0.25)
		continue
	env_img = np.zeros((SIZE+BORDER,SIZE+BORDER,3), np.uint8)
	#grid_img = np.array(log_prob_maps[idx] * 255, dtype=np.uint8)

	# cv2.rectangle(img, scale_point(Field.dz_square['bl']), 
	# 			  scale_point(Field.dz_square['tr']), 
	# 			  (0,0,255), -1)

	for wall in Field.walls:
		cv2.line(env_img, scale_point(wall[0]), 
					scale_point(wall[1]), 
					(0,0,255), 1)

	for lidar_intersect in all_visual_intersect_data[idx]:
		cv2.circle(env_img,scale_point(lidar_intersect),1,(0,255,0),-1)

	for landmark in landmarks[idx]:
		cv2.circle(env_img,scale_point(landmark),5,(255,0,0),-1)

	for particle in all_particles_pos[idx]:
		cv2.circle(env_img,scale_point(particle.getPos()),1,(100,100,100),-1)

	cv2.circle(env_img,scale_point(best_particle_pos[idx]),5,(0,255,255),-1)

	cv2.circle(env_img,scale_point(ground_truth[idx][0:2]),5,(0,255,0),-1)
	# orientation
	cv2.line(env_img, scale_point(ground_truth[idx])[0:2],
		scale_point((ground_truth[idx][0]+(math.cos(ground_truth[idx][2])*10), ground_truth[idx][1]+(math.sin(ground_truth[idx][2])*10))),
		(255,0,0), 1)
	
	cv2.putText(env_img,"Landmarks",(30,20),1,1.0,(0,0,0), 2)
	cv2.putText(env_img,"Landmarks",(30,20),1,1.0,(255,0,0), 1)
	
	cv2.putText(env_img,"Particles",(30,40),1,1.0,(0,0,0), 2)
	cv2.putText(env_img,"Particles",(30,40),1,1.0,(100,100,100), 1)
	
	cv2.putText(env_img,"Best Particle",(30,60),1,1.0,(0,0,0), 2)
	cv2.putText(env_img,"Best Particle",(30,60),1,1.0,(0,255,255), 1)
	
	cv2.putText(env_img,"Ground Truth",(30,80),1,1.0,(0,0,0), 2)
	cv2.putText(env_img,"Ground Truth",(30,80),1,1.0,(0,255,0), 1)
	
	cv2.putText(env_img, "Obstacle", (30, 100),1,1.0,(0,0,0), 2)
	cv2.putText(env_img, "Obstacle", (30, 100),1,1.0,(0,0,255), 1)

	cv2.imshow("Particle Filter", env_img)
	
	plt.clf()
	pose = ground_truth[idx]
	circle = plt.Circle((pose[1], pose[0]), radius=3.0, fc='y')
	plt.gca().add_patch(circle)
	arrow = pose[0:2] + np.array([3.5, 0]).dot(np.array([[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]))
	plt.plot([pose[1], arrow[1]], [pose[0], arrow[0]])
	plt.imshow(1.0 - 1./(1.+np.exp(log_prob_maps[idx])), 'Greys')
	plt.pause(0.00005)

	idx += 1
	time.sleep(0.25)