import numpy as np
import scipy.io
import scipy.stats
import random

import constants

class Occupancy_Grid():
    def __init__(self, xsize, ysize, grid_size):
        self.xsize =  xsize+2
        self.ysize = ysize+2
        self.grid_size = grid_size

        self.log_prob_map = np.zeros((self.xsize, self.ysize))

        self.alpha = 1.0 # thickness of obstacles
        self.beta = constants.LIDAR_WIDTH # width of beam
        self.max_range = constants.LIDAR_RANGE

        self.grid_position_m = np.array([
            np.tile(np.arange(0, self.xsize*self.grid_size, self.grid_size)[:, None], (1, self.ysize)),
            np.tile(np.arange(0, self.ysize*self.grid_size, self.grid_size)[:,None].T, (self.xsize, 1))
        ])

        # log probs
        self.l_occ = np.log(0.65/0.35)
        self.l_free = np.log(0.35/0.65)

        self.max_landmarks = 5
        self.landmarks = []
        #self.landmarks = [[10,50], [20, 50], [30,50], [5, 10], [5, 20], [5, 50], [10, 5], [20, 5], [30, 5], [40, 5], [95, 50], [50, 5], [60, 15], [70, 25]]

    def update_map(self, pose, z):
        # normalize pose orientation
        if pose[2] > np.pi:
            pose[2] -= 2*np.pi

        if pose[2] < -np.pi:
            pose[2] += 2*np.pi

        dx = self.grid_position_m.copy()
        dx[0, :, :] -= pose[0]
        dx[1, :, :] -= pose[1]
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) # matrix of all angles relative to racecar

        # wrap +pi to -pi
        theta_to_grid[theta_to_grid > np.pi] -= 2*np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2*np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0)

        # each laser beam (range, bearing)
        for z_i in z:
            r = z_i[0]
            b = z_i[1]

            if r >= self.max_range:
                continue

            free_mask = (np.abs(theta_to_grid -b) <= self.beta/2.0) & (dist_to_grid < (r - self.alpha/2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.beta/2.0) & (np.abs(dist_to_grid - r) <= self.alpha/2.0)

            # adjust map
            self.log_prob_map[free_mask] += self.l_free
            self.log_prob_map[occ_mask] += self.l_occ

        self.landmarks = []
        landmark_mask = (np.abs(theta_to_grid-pose[2]) < np.pi/2) 
        max_idxs = np.argpartition((self.log_prob_map*landmark_mask).flatten(), -self.max_landmarks)[-self.max_landmarks:]
        for max_idx in max_idxs:
            max_x = int(max_idx / len(self.log_prob_map))
            max_y = max_idx % len(self.log_prob_map)

            max_angle = np.arctan2(max_y - pose[1], max_x - pose[0])
            if max_angle > np.pi:
                max_angle -= 2*np.pi

            if max_angle < -np.pi:
                max_angle += 2*np.pi

            # if np.abs(max_angle-pose[2]) >= np.pi/4: # figure out why this is happening (not supposed to get any max behind bot)
            #     continue

            if self.log_prob_map[max_x][max_y] <= 0.5:
                continue
            self.landmarks.append([max_x, max_y])

        
        