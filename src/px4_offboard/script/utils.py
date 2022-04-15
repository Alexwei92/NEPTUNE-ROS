import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

BLACK  = (0,0,0)
WHITE  = (255,255,255)
RED    = (0,0,255)
BLUE   = (255,0,0)
GREEN  = (0,255,0)
YELLOW = (0,255,255)

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def wrap_2PI(radian):
    res = radian % (math.pi * 2.0)
    if (res < 0):
        res = res +math.pi * 2.0
    return res

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
            'lower': pos_lower}

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

def calculate_affordance(map_data, pilot_pose):
    '''
    Calculate affordance
    '''
    dist_center = np.array([])
    dist_left   = np.array([])
    dist_right  = np.array([])
    rel_angle   = np.array([])
    in_river    = np.array([])

    pilot_pos_x, pilot_pos_y = pilot_pose['pos'][0], pilot_pose['pos'][1]
    pilot_yaw = pilot_pose['yaw']
    direction = pilot_pose['direction']

    pos_x_center, pos_y_center  = map_data['center'][:,0], map_data['center'][:,1]
    pos_x_norm, pos_y_norm      = map_data['norm'][:,0], map_data['norm'][:,1]
    pos_x_tang, pos_y_tang      = map_data['tang'][:,0], map_data['tang'][:,1]
    pos_x_upper, pos_y_upper    = map_data['upper'][:,0], map_data['upper'][:,1]
    pos_x_lower, pos_y_lower    = map_data['lower'][:,0], map_data['lower'][:,1]   
    total_N = len(pos_x_center)
    
    index = np.argmin(np.sqrt((pilot_pos_x - pos_x_center)**2 + (pilot_pos_y - pos_y_center)**2))

    # 1) Distance to center
    a = (pilot_pos_y-pos_y_center[index]) / abs(pilot_pos_y-pos_y_center[index]) * direction
    dist_center_tmp = a*np.sqrt((pilot_pos_x - pos_x_center[index])**2 + (pilot_pos_y - pos_y_center[index])**2)
    dist_center = np.append(dist_center, dist_center_tmp)

    # 2) Distance to left and right
    window_size = 20
    search_range = range(max(1, math.ceil(index - window_size / 2)), min(total_N, math.floor(index + window_size / 2)))

    if direction > 0:
        x_left, y_left = pos_x_upper[search_range], pos_y_upper[search_range]
        x_right, y_right = pos_x_lower[search_range], pos_y_lower[search_range]
    else:   
        x_left, y_left = pos_x_lower[search_range], pos_y_lower[search_range]
        x_right, y_right = pos_x_upper[search_range], pos_y_upper[search_range]
    
    line_vector = np.asarray([[pos_x_center[index], pos_y_center[index]], [pos_x_norm[index], pos_y_norm[index]]])
    left_index = find_min_dist_line2points(line_vector,  np.column_stack([x_left, y_left]))
    right_index = find_min_dist_line2points(line_vector, np.column_stack([x_right, y_right]))

    dist_left = np.append(dist_left,
                np.sqrt((pilot_pos_x - x_left[left_index])**2 + (pilot_pos_y - y_left[left_index])**2))
    dist_right = np.append(dist_right,
                np.sqrt((pilot_pos_x - x_right[right_index])**2 + (pilot_pos_y - y_right[right_index])**2))

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
    if direction > 0:
        angle += np.pi
    angle_diff = pilot_yaw - angle
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

def plot_vehicle(handle, pos, heading, show_FOV=True, is_first=False):
    '''
    plot the vehicle with FOV
    '''
    FOV = 70 * np.pi / 180
    # location
    if is_first:
        origin, = handle.plot(pos[0], pos[1], marker='o', markersize=4, fillstyle='full', color='r')
    else:
        handle['origin'].set_xdata(pos[0])
        handle['origin'].set_ydata(pos[1])
    # heading
    L = 2
    forward_x = [pos[0], pos[0] + L*np.cos(heading)]
    forward_y = [pos[1], pos[1] + L*np.sin(heading)]
    right_x = [pos[0], pos[0] + L*np.cos(heading+np.pi/2)]
    right_y = [pos[1], pos[1] + L*np.sin(heading+np.pi/2)]
    if is_first:
        forward_line, = handle.plot(forward_x, forward_y, color='b', linewidth=2)
        right_line, = handle.plot(right_x, right_y, color='g', linewidth=2)
    else:
        handle['forward_line'].set_xdata(forward_x)
        handle['forward_line'].set_ydata(forward_y)
        handle['right_line'].set_xdata(right_x)
        handle['right_line'].set_ydata(right_y)

    if show_FOV:
        L = 5 / np.cos(FOV/2)
        left_point = [pos[0] + L*np.cos(heading+FOV/2), pos[1] + L*np.sin(heading+FOV/2)]
        right_point = [pos[0] + L*np.cos(heading-FOV/2), pos[1] + L*np.sin(heading-FOV/2)]
        path = [pos, left_point, right_point]
        if is_first:
            FOV_patch = handle.add_patch(patches.Polygon(path, color='w', linestyle='', alpha=0.25))
        else:
            handle['FOV_patch'].set_xy(path)

    if is_first:
        return {'origin': origin,
                'forward_line': forward_line,
                'right_line': right_line, 
                'FOV_patch': FOV_patch}   

class MapPlot():
    '''
    plot the map of the environment
    '''
    def __init__(self, map_path):
        self.fig, self.axes = plt.subplots()
        self.map_data = read_map_data(map_path)
        self.initialize_map()
        self.has_initialized = False
        self.start_point = None
        self.end_point = None

    def initialize_map(self, disp_intervel=10):
        self.axes.plot(self.map_data['center'][:,0], self.map_data['center'][:,1], color='w', linewidth=0.5)
        self.axes.plot(self.map_data['upper'][:,0], self.map_data['upper'][:,1], color=(0.9290, 0.6940, 0.1250), linewidth=2.0)
        self.axes.plot(self.map_data['lower'][:,0], self.map_data['lower'][:,1], color=(0.9290, 0.6940, 0.1250), linewidth=2.0)
        self.axes.set_aspect('equal')
        self.axes.grid(alpha=0.15)
        x_range, y_range = self.axes.get_xlim(), self.axes.get_ylim()
        x_range = (disp_intervel*math.floor(x_range[0]/disp_intervel), disp_intervel*math.ceil(x_range[1]/disp_intervel))
        y_range = (disp_intervel*math.floor(y_range[0]/disp_intervel), disp_intervel*math.ceil(y_range[1]/disp_intervel))
        self.axes.set_xticks(range(x_range[0], x_range[1], disp_intervel))
        self.axes.set_xticklabels([])
        self.axes.set_yticks(range(y_range[0], y_range[1], disp_intervel))
        self.axes.set_yticklabels([])
        self.axes.tick_params(direction='out', length=0, color='w')
        self.axes.set_xlabel('x', fontsize=16)
        self.axes.set_ylabel('y', fontsize=16)
        self.fig.tight_layout()

    def update_start_point(self, start_point):
        start_upper_idx = np.abs(self.map_data['upper'][:,0] - start_point[0]).argmin()
        start_lower_idx = np.abs(self.map_data['lower'][:,0] - start_point[0]).argmin()

        self.axes.plot([self.map_data['upper'][start_upper_idx,0], self.map_data['lower'][start_lower_idx,0]], 
                        [self.map_data['upper'][start_upper_idx,1], self.map_data['lower'][start_lower_idx,1]], 
                        color='w', linestyle='-.', linewidth=1)

        self.axes.plot(start_point[0], start_point[1], marker='o', markersize=10, fillstyle='none', color='g')
        # self.axes.add_patch(plt.Circle(start_point, 2, color='r', alpha=0.5))

    def update_end_point(self, end_point):
        end_upper_idx = np.abs(self.map_data['upper'][:,0] - end_point[0]).argmin()
        end_lower_idx = np.abs(self.map_data['lower'][:,0] - end_point[0]).argmin()

        self.axes.plot([self.map_data['upper'][end_upper_idx,0], self.map_data['lower'][end_lower_idx,0]], 
                        [self.map_data['upper'][end_upper_idx,1], self.map_data['lower'][end_lower_idx,1]], 
                        color='w', linestyle='-.', linewidth=1)

        self.axes.plot(end_point[0], end_point[1], marker='o', markersize=5, fillstyle='none', color='r')
        # self.axes.add_patch(plt.Circle(end_point, 2, color='c', alpha=0.5))

    def update(self, pos, heading):
        if self.has_initialized:
            plot_vehicle(self.axes_dict, pos, heading, is_first=False)
        else:
            self.axes_dict = plot_vehicle(self.axes, pos, heading, is_first=True)
            self.has_initialized = True

    def reset(self):
        self.axes.clear()
        self.initialize_map()
        self.has_initialized = False
        self.start_point = None
        self.end_point = None