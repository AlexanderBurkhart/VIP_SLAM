from robot import Robot
import random
import numpy as np

class ParticleFilter:
    def __init__(self, agent, N=1000):
        self.p = []
        self.last_best_p = None
        self.N = N

        self.noises = {
            'forward': 0.05,
            'turn': 0.05,
            'sense': 5.0
        }

        for _ in range(N):
            x = Robot(specPos=agent.getPos())
            x.set_noise(self.noises['forward'], self.noises['turn'], self.noises['sense'])
            self.p.append(x)

    def step(self, occupancy_grid, dists, agent, speed):
        #TODO: for the map never assume that the particles is completely correct
        multiplier = 3
        p1 = []
        # 80% samples around last position
        for i in range(0, self.N):#self.N):
            p1.append(self.p[i].move(random.random() * speed * multiplier, random.random() * 2 * np.pi, ignore_walls=True))
            # p1.append(self.p[i].move(random.random() * speed * multiplier, agent.orientation, ignore_walls=True))

        self.p = p1

        w = []
        for i in range(self.N):
            w.append(self.p[i].measurement_prob(dists, occupancy_grid))

        # rsr change
        w_sum = sum(w)
        w = [weight / w_sum for weight in w]

        # p3 = []
        # index = int(random.random() * self.N)
        # beta = 0.0
        # mw = max(w)
        # for i in range(self.N):
        #     beta += random.random() * 2.0 * mw
        #     while beta > w[index]:
        #         beta -= w[index]
        #         index = (index + 1) % self.N
        #     p3.append(self.p[index])
        # self.p = p3
        
        # RSR
        p3 = []
        i = 0
        u = random.random() / self.N
        j = 0
        while j < len(w):
            Ns = np.floor(self.N * (w[j]-u)) + 1
            counter = 1
            while counter <= Ns:
                p3.append(self.p[j])
                counter += 1
            u += Ns/self.N-w[j]
            j += 1

        self.p = p3

        # weighted average instead
        weight_sum = 0
        position_sum = np.array([0.0, 0.0])
        for i in range(len(self.p)):
            p_weight = self.p[i].measurement_prob(dists, occupancy_grid)
            weight_sum += p_weight
            p_state = self.p[i].get_state()
            position_sum[0] += p_weight * p_state[0]
            position_sum[1] += p_weight * p_state[1]
        
        best_p_pos = position_sum / weight_sum
        best_p = Robot(specPos=best_p_pos, specOrientation=agent.orientation)
        best_p.set_noise(self.noises['forward'], self.noises['turn'], self.noises['sense'])
        

        # check if new position realistically makes sense
        # check if distance exceeds given speed
        if self.last_best_p:
            best_p_pos = best_p.get_state()[0:2]
            last_best_p_pos = self.last_best_p.get_state()[0:2]
            if np.linalg.norm(best_p_pos - last_best_p_pos) > speed/5:
                # find angle and step towards that direction from speed
                angle = np.arctan2(best_p_pos[1]-last_best_p_pos[1], best_p_pos[0]-last_best_p_pos[0])
                best_p = self.last_best_p
                best_p.x += np.cos(angle) * speed
                best_p.y += np.sin(angle) * speed

        self.last_best_p = best_p
        # num_p = len(self.p)
        # best_p = None
        # highest_weight = -1.0

        # for i in range(num_p):
        #     p_weight = self.p[i].measurement_prob(dists, occupancy_grid)
        #     if p_weight > highest_weight:
        #         highest_weight = p_weight
        #         best_p = self.p[i]
        
        return best_p, self.p