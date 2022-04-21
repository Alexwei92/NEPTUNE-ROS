import math
import numpy as np

from math_utils import find_min_dist_line2points, wrap_PI

def calculate_affordance(map_data, pose):
    '''
    Calculate affordance
    '''
    pilot_pos_x, pilot_pos_y = pose['pos'][0], pose['pos'][1]
    pilot_yaw = pose['yaw']
    direction = pose['direction']

    pos_x_center, pos_y_center  = map_data['center'][:,0], map_data['center'][:,1]
    pos_x_norm, pos_y_norm      = map_data['norm'][:,0], map_data['norm'][:,1]
    pos_x_tang, pos_y_tang      = map_data['tang'][:,0], map_data['tang'][:,1]
    pos_x_upper, pos_y_upper    = map_data['upper'][:,0], map_data['upper'][:,1]
    pos_x_lower, pos_y_lower    = map_data['lower'][:,0], map_data['lower'][:,1]   
    total_N = len(pos_x_center)
    
    # Closest point index along the centerline
    index = np.argmin(np.sqrt((pilot_pos_x - pos_x_center)**2 + (pilot_pos_y - pos_y_center)**2))

    # 1) Distance to center
    a = -(pilot_pos_y - pos_y_center[index]) / abs(pilot_pos_y - pos_y_center[index]) * direction
    dist_center = a * np.sqrt((pilot_pos_x - pos_x_center[index])**2 + (pilot_pos_y - pos_y_center[index])**2)

    # 2) Distance to left and right
    window_size = 30 # check [-window_size/2, window_size/2] neighbour points
    search_range = range(max(1, math.ceil(index - window_size / 2)), min(total_N, math.floor(index + window_size / 2)))

    if direction > 0:
        x_left,  y_left  = pos_x_upper[search_range], pos_y_upper[search_range]
        x_right, y_right = pos_x_lower[search_range], pos_y_lower[search_range]
    else:   
        x_left,  y_left  = pos_x_lower[search_range], pos_y_lower[search_range]
        x_right, y_right = pos_x_upper[search_range], pos_y_upper[search_range]
    
    line_vector = np.asarray([[pos_x_center[index], pos_y_center[index]], [pos_x_norm[index], pos_y_norm[index]]])
    left_index  = find_min_dist_line2points(line_vector, np.column_stack([x_left, y_left]))
    right_index = find_min_dist_line2points(line_vector, np.column_stack([x_right, y_right]))

    dist_left = np.sqrt((pilot_pos_x - x_left[left_index])**2 + (pilot_pos_y - y_left[left_index])**2)
    dist_right = np.sqrt((pilot_pos_x - x_right[right_index])**2 + (pilot_pos_y - y_right[right_index])**2)

    dist_left_max  = np.sqrt((pos_x_center[index] - x_left[left_index])**2 + (pos_y_center[index] - y_left[left_index])**2)
    dist_right_max = np.sqrt((pos_x_center[index] - x_right[right_index])**2 + (pos_y_center[index] - y_right[right_index])**2)
    
    # 3) Is in the bounded region
    if dist_center < 0 and -dist_center > dist_left_max:
        in_bound = False
    elif dist_center > 0 and dist_center > dist_right_max:
        in_bound = False
    elif pilot_pos_x < pos_x_center[0] or pilot_pos_x > pos_x_center[-1]:
        in_bound = False
    else:
        in_bound = True

    # 4) Angle in radian
    angle = np.arctan2(pos_y_tang[index] - pos_y_center[index], pos_x_tang[index] - pos_x_center[index])
    if direction < 0:
        angle += np.pi
        
    angle_diff = angle - pilot_yaw
    rel_angle = wrap_PI(angle_diff)

    return {'dist_center':  dist_center,
            'dist_left':    dist_left,
            'dist_right':   dist_right,
            'rel_angle':    rel_angle,
            'in_bound':     in_bound
            }