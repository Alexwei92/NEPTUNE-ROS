import numpy as np
import math
import pandas
import utm
from scipy.interpolate import CubicSpline
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utils.math_utils import constrain_value

""" Spline Smoothing """
class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        # print(np.diff(self.s))
        singular_idx = np.where(np.diff(self.s) <= 0)[0]
        if len(singular_idx) > 0:
            print("[WARNING]: non-increasing indexes are detected!!!!")
            self.s = np.delete(self.s, singular_idx, axis=0)
            x = np.delete(x, singular_idx, axis=0)
            y = np.delete(y, singular_idx, axis=0)

        self.sx = CubicSpline(self.s, x)
        self.sy = CubicSpline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))

        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx(s)
        y = self.sy(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = np.asarray(self.sx(s, 1))
        dy = np.asarray(self.sy(s, 1))
        ddx = np.asarray(self.sx(s, 2))
        ddy = np.asarray(self.sy(s, 2))

        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3.0 / 2.0))
        # print("k: ", k)
        return k

    def calc_curvature_prime(self, s):
        """
        calc curvature
        """
        dx = self.sx(s, 1)
        dy = self.sy(s, 1)
        ddx = self.sx(s, 2)
        ddy = self.sy(s, 2)
        dddx = self.sx(s, 3)
        dddy = self.sy(s, 3)

        p1 = dx ** 2 + dy ** 2
        p2 = dx * dddy - dy * dddx
        p3 = dx * ddy - ddx * dy
        p4 = dx * ddx + dy * ddy
        bottom = (dx ** 2 + dy ** 2) ** (5 / 2)
        k_prime = (p1 * p2 - 3 * p3 * p4) / bottom

        return k_prime

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx(s, 1)
        dy = self.sy(s, 1)
        yaw = np.arctan2(dy, dx)

        return yaw


def calc_spline_course(x, y, ds=0.1):
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


""" Path Configuring """
def get_projection_point(x_m, y_m, yaw_m, k_m, x, y):

    # get projection point on the curve
    d_vector = np.array([x - x_m, y - y_m])
    tau_vector = np.array([math.cos(yaw_m), math.sin(yaw_m)])
    p_proj = np.array([x_m, y_m]) + (d_vector.dot(tau_vector)) * tau_vector
    yaw_proj = yaw_m + k_m * (d_vector.dot(tau_vector))

    return p_proj, yaw_proj

def read_map_data(filepath):
    """
    Read map spline data
    """
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


""" Affordance Contorl """
def calc_affordance_cmd(affordance, max_yawrate=45):
    if affordance is None:
        return 0.0, None

    if 'dist_center_width' in affordance:
        dist_center_width = affordance['dist_center_width'] 
        dist_left_width = affordance['dist_left_width'] - 0.5
    else:
        dist_center_width = affordance['dist_center'] / (affordance['dist_left'] + affordance['dist_right'])
        dist_left_width = affordance['dist_left'] / (affordance['dist_left'] + affordance['dist_right']) - 0.5
    
    rel_angle = affordance['rel_angle']

    # if abs(rel_angle) < 3 / 180 * math.pi:
    #     rel_angle = 0

    # if abs(rel_angle) < 0.03:
    #     dist_center_width = 0

    # Option 1: Sigmoid function
    cmd = 1.0 * (2 /(1 + math.exp(15*(1.5*rel_angle/(math.pi/2) + 1.0*dist_center_width))) - 1)
    
    # Option 2: Stanley
    # stanley_output = rel_angle + math.atan(2.5 * affordance['dist_center'] / 1.5)
    # cmd = stanley_output * (15) / max_yawrate
    # cmd = -cmd

    return constrain_value(cmd, -1.0, 1.0), affordance['in_bound']


def get_local_xy_from_latlon(lat, lon, utm_T_local):
    e, n, _, _ = utm.from_latlon(lat, lon)
    utm_pose = np.array([e, n, 0, 1]).T
    local_pose = utm_T_local.dot(utm_pose)
    local_pos = local_pose[:3].T

    return local_pos

def get_projection_point2line(point, line):
    p1_p2 = line[1] - line[0]
    p1_q = point[:2] - line[0]
    lateral_proj = np.cross(p1_q, p1_p2 / np.linalg.norm(p1_p2))  # positive means on the right hand side
    longitude_proj = np.dot(p1_q, p1_p2) / np.linalg.norm(p1_p2)

    return lateral_proj, longitude_proj

def find_area_index(current_pose, all_data, start_index=0):
    search_index = np.arange(0, len(all_data))
    search_index_order = (search_index + start_index) % len(all_data)

    for index in search_index_order:
        point = Point(current_pose[0], current_pose[1])
        polygon = Polygon([vertex for vertex in all_data[index]['vertice']])

        if polygon.contains(point):
            return index

    return None

def whether_in_polygon(current_pose, polygon):
    point = Point(current_pose[0], current_pose[1])

    if polygon.contains(point):
        return True
    else:
        return False