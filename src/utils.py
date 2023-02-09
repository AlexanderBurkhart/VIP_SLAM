import numpy as np

def softmax(x):
    '''
        Softmax function
    '''
    x = np.exp(x-np.max(x))
    return x / x.sum()

def polar2cart(points, angles):
    '''
        Transform points from polar to catesian coordinate
        
        Input:
            points - point distance measured from lidar
            angles - lidar scan range, from -135° to 135°
        Outputs:
            x - x coordinate of points
            y - y coordinate of points
    '''
    x, y = points * np.cos(angles), points * np.sin(angles)
    return x, y