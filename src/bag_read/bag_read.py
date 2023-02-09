import pandas as pd
import numpy as np
import constants
import cv2
import ast
#from cv_bridge import CvBridge

all_speed_data = {}
all_scan_data = {}
all_gt_data = {}
all_imu_data = {}
all_vision_data = {}

lidar_data_start = 61 # reading 15 deg to 270 deg
lidar_data_end = 1080

def get_nearest_time(times, timestep):
    cur_time = None
    for i in range(0, len(times)):
        time = times.iloc[i]
        if time > timestep:
            if cur_time:
                if abs(cur_time-timestep) < abs(time-timestep):
                    return i-1
                return i
            else:
                raise Exception("no timestep valid")
        cur_time = time


def get_times(bag_name):
    # not the best way, speed had the lowest amount of timesteps, so using it for all timesteps
    if bag_name not in all_speed_data:
        speed_data = pd.read_csv('bag_read/'+bag_name+'/ackermann_cmd_mux-input-teleop.csv')
        all_speed_data[bag_name] = speed_data
    else:
        speed_data = all_speed_data[bag_name]
    return speed_data['Time']

# TODO: USE TIME FOR INDEXING AND NOT TIMESTEP
def read_speed(bag_name, timestep):
    if bag_name not in all_speed_data:
        speed_data = pd.read_csv('bag_read/'+bag_name+'/ackermann_cmd_mux-input-teleop.csv')
        all_speed_data[bag_name] = speed_data
    else:
        speed_data = all_speed_data[bag_name]
    idx = get_nearest_time(speed_data['Time'], timestep)
    if idx >= len(speed_data):
        return None
    return speed_data['drive.speed'][idx]

def read_lidar(bag_name, timestep, orientation):
    if bag_name not in all_scan_data:
        scan_data = pd.read_csv('bag_read/'+bag_name+'/scan.csv')
        all_scan_data[bag_name] = scan_data
    else:
        scan_data = all_scan_data[bag_name]
    idx = get_nearest_time(scan_data['Time'], timestep)
    if idx >= len(scan_data):
        return None
    ranges_needed = ['ranges_%i' % r for r in range(lidar_data_start,lidar_data_end+1)]
    lidar_data = pd.DataFrame({'range': scan_data.loc[idx, ranges_needed]})
    lidar_data['angle'] = [(constants.LIDAR_WIDTH * (i+lidar_data_start)) - np.pi + orientation for i in range(0, len(ranges_needed))]
    return lidar_data.to_numpy()

def read_ground_truth(bag_name, timestep):
    if bag_name not in all_gt_data:
        gt_data = pd.read_csv('bag_read/'+bag_name+'/racecar-vicon_position.csv')
        gt_data['pose.position.x'] += abs(min(gt_data['pose.position.x']))+10
        gt_data['pose.position.y'] += abs(min(gt_data['pose.position.y']))+10
        all_gt_data[bag_name] = gt_data
    else:
        gt_data = all_gt_data[bag_name]
    idx = get_nearest_time(gt_data['Time'], timestep)
    if idx >= len(gt_data):
        return None
    return gt_data['pose.position.x'][idx], gt_data['pose.position.y'][idx], read_imu(bag_name, timestep)

def read_imu(bag_name, timestep):
    # NOTE: imu data seems weird using direction of velocity vector
    # if bag_name not in all_imu_data:
    #     imu_data = pd.read_csv('bag_read/'+bag_name+'/imu-data.csv')
    #     all_imu_data[bag_name] = imu_data
    # else:
    #     imu_data = all_imu_data[bag_name]
    # idx = get_nearest_time(imu_data['Time'], timestep)
    # return (imu_data['orientation.x'][idx]*np.pi)

    if bag_name not in all_gt_data:
        gt_data = pd.read_csv('bag_read/'+bag_name+'/racecar-vicon_position.csv')
        gt_data['pose.position.x'] += abs(min(gt_data['pose.position.x']))+10
        gt_data['pose.position.y'] += abs(min(gt_data['pose.position.y']))+10
        all_gt_data[bag_name] = gt_data
    else:
        gt_data = all_gt_data[bag_name]
    idx = get_nearest_time(gt_data['Time'], timestep)
    if idx >= len(gt_data):
        return None
    cur_pos = gt_data['pose.position.x'][idx], gt_data['pose.position.y'][idx]
    prev_pos = gt_data['pose.position.x'][idx-1], gt_data['pose.position.y'][idx-1]

    return np.arctan2(cur_pos[1]-prev_pos[1], cur_pos[0]-prev_pos[0])

def read_vision(bag_name, timestep):
    if bag_name not in all_vision_data:
        vision_data = pd.read_csv('bag_read/'+bag_name+'/vision.csv')
        all_vision_data[bag_name] = vision_data
    else:
        vision_data = all_vision_data[bag_name]
    idx = get_nearest_time(vision_data['Time'], timestep)+15
    raw_string = vision_data.iloc[idx]['data']
    byte_string = raw_string[2:-1].encode('latin1')
    escaped_string = byte_string.decode('unicode_escape')
    byte_string = escaped_string.encode('latin1')
    nparr = np.fromstring(byte_string, np.uint8)
    rgb = nparr.reshape((360, 640, -1))
    return rgb