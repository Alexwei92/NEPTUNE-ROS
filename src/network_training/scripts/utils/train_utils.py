import yaml
import pandas
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def read_yaml(file_path):
    '''
    Read yaml file
    '''
    try:
        file = open(file_path, 'r')
        config = yaml.safe_load(file)
        file.close()
        return config

    except Exception as error:
        print(error)
        exit('Exit the program.')


def read_map_data(filepath):
    '''
    Read map spline data
    '''
    spline_data = pandas.read_csv(filepath)
    pos_center = spline_data[['pos_x_center', 'pos_y_center']].to_numpy()
    pos_upper = spline_data[['pos_x_upper', 'pos_y_upper']].to_numpy()
    pos_lower = spline_data[['pos_x_lower', 'pos_y_lower']].to_numpy()
    pos_seg_num = spline_data['pos_seg_num'].to_numpy()

    # Calculate Normal Vector
    # x1x2 + y1y2 = 0 if two lines are perpendicular
    L = 10
    tangent = np.gradient(pos_center[:,1], pos_center[:,0])
    tang_dx = L / np.sqrt(1 + tangent**2)
    tang_dy = tangent * L / np.sqrt(1 + tangent**2)
    pos_x_tang = pos_center[:,0] + tang_dx
    pos_y_tang = pos_center[:,1] + tang_dy
    c = tang_dx / tang_dy
    pos_y_norm = pos_center[:,1] + np.sqrt(c**2 * L**2 / (c**2 +1)) 
    pos_x_norm = pos_center[:,0] - np.sqrt(c**2 * L**2 / (c**2 +1)) / c
    pos_tang = np.column_stack([pos_x_tang, pos_y_tang])
    pos_norm = np.column_stack([pos_x_norm, pos_y_norm])

    return {'center': pos_center,
            'tang': pos_tang,
            'norm': pos_norm, 
            'upper': pos_upper,
            'lower': pos_lower, 
            'seg_num': pos_seg_num}

def read_pilot_data(filepath):
    '''
    Read airsim file data
    '''
    data = pandas.read_csv(filepath)
    pilot_pos = data[['pos_x','pos_y','pos_z']][:-1].to_numpy()
    pilot_yaw = data['yaw'][:-1].to_numpy()
    direction = 1 if pilot_pos[-1,0] > pilot_pos[0,0] else -1
    return {'pos': pilot_pos,
            'yaw': pilot_yaw,
            'direction': direction}

def find_min_dist_line2points(query_line, Points):
    '''
    Distance from a line (i.e., x1,y1,x2,y2) to a series of points
    
    dist = |A*x+B*y+C| / (A^2+B^2)
    where A = y1-y2
          B = x2-x1
          C = x1*y2-x2*y1
    '''
    A = query_line[0,1] - query_line[1,1]
    B = query_line[1,0] - query_line[0,0]
    C = query_line[0,0] * query_line[1,1] - query_line[1,0] * query_line[0,1]
    dist = abs(A*Points[:,0] + B*Points[:,1] + C) / (A**2 + B**2)
    return np.argmin(dist)

def calculate_affordance(map_data, pilot_data):
    '''
    Calculate affordance
    '''
    dist_center = np.array([])
    dist_left = np.array([])
    dist_right = np.array([])
    rel_angle = np.array([])
    in_river = np.array([])

    pilot_pos_x, pilot_pos_y = pilot_data['pos'][:,0], pilot_data['pos'][:,1]
    pilot_yaw = pilot_data['yaw']
    direction = pilot_data['direction']

    pos_x_center, pos_y_center = map_data['center'][:,0], map_data['center'][:,1]
    pos_x_norm, pos_y_norm = map_data['norm'][:,0], map_data['norm'][:,1]
    pos_x_tang, pos_y_tang = map_data['tang'][:,0], map_data['tang'][:,1]
    pos_x_upper, pos_y_upper = map_data['upper'][:,0], map_data['upper'][:,1]
    pos_x_lower, pos_y_lower = map_data['lower'][:,0], map_data['lower'][:,1]   
    pos_seg_num = map_data['seg_num'] 
    
    for i in range(len(pilot_pos_x)):
        index = np.argmin(np.sqrt((pilot_pos_x[i] - pos_x_center)**2 + (pilot_pos_y[i] - pos_y_center)**2))

        # 1) Distance to center
        a = (pilot_pos_y[i]-pos_y_center[index]) / abs(pilot_pos_y[i]-pos_y_center[index]) * direction
        dist_center_tmp = a*np.sqrt((pilot_pos_x[i] - pos_x_center[index])**2 + (pilot_pos_y[i] - pos_y_center[index])**2)
        dist_center = np.append(dist_center, dist_center_tmp)

        # 2) Distance to left and right
        search_range = (pos_seg_num == pos_seg_num[index])
        if direction < 0: # remember y-axis is reversed
            x_left, y_left = pos_x_upper[search_range], pos_y_upper[search_range]
            x_right, y_right = pos_x_lower[search_range], pos_y_lower[search_range]
        else:   
            x_left, y_left = pos_x_lower[search_range], pos_y_lower[search_range]
            x_right, y_right = pos_x_upper[search_range], pos_y_upper[search_range]
        
        line_vector = np.asarray([[pos_x_center[index], pos_y_center[index]], [pos_x_norm[index], pos_y_norm[index]]])
        left_index = find_min_dist_line2points(line_vector,  np.column_stack([x_left, y_left]))
        right_index = find_min_dist_line2points(line_vector, np.column_stack([x_right, y_right]))

        dist_left = np.append(dist_left,
                    np.sqrt((pilot_pos_x[i] - x_left[left_index])**2 + (pilot_pos_y[i] - y_left[left_index])**2))
        dist_right = np.append(dist_right,
                    np.sqrt((pilot_pos_x[i] - x_right[right_index])**2 + (pilot_pos_y[i] - y_right[right_index])**2))

        dist_left_center = np.sqrt((pos_x_center[index] - x_left[left_index])**2 + (pos_y_center[index] - y_left[left_index])**2)
        dist_right_center = np.sqrt((pos_x_center[index] - x_right[right_index])**2 + (pos_y_center[index] - y_right[right_index])**2)
        
        is_valid = True
        if dist_center_tmp < 0 and -dist_center_tmp > dist_left_center:
            is_valid = False
        elif dist_center_tmp > 0 and dist_center_tmp > dist_right_center:
            is_valid = False
        in_river = np.append(in_river, is_valid)

        # 4) Angle
        angle = np.arctan2(pos_y_tang[index]-pos_y_center[index], pos_x_tang[index]-pos_x_center[index])
        if direction < 0:
            angle += np.pi
        angle_diff = pilot_yaw[i] - angle
        if angle_diff > np.pi:
            angle_diff -= np.pi*2
        if angle_diff < -np.pi:
            angle_diff += np.pi*2
        rel_angle = np.append(rel_angle, angle_diff)

    if len(dist_center) == 1:
        dist_center = dist_center[0]
        dist_left   = dist_left[0]
        dist_right  = dist_right[0]
        rel_angle   = rel_angle[0]
        in_river    = in_river[0]

    return {'dist_center': dist_center,
            'dist_left': dist_left,
            'dist_right': dist_right,
            'rel_angle': rel_angle,
            'in_river': in_river}