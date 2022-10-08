import numpy as np
import utm

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def get_projection_point2line(point, line):
    p1_p2 = line[1] - line[0]
    p1_q = point[:2] - line[0]
    lateral_proj = np.cross(p1_q, p1_p2 / np.linalg.norm(p1_p2))  # positive means on the right hand side
    longitude_proj = np.dot(p1_q, p1_p2)

    return lateral_proj, longitude_proj

def find_area_index(current_pose, all_data, frame='local', start_index=0):
    search_index = np.arange(0, len(all_data))
    search_index_order = (search_index + start_index) % len(all_data)

    for index in search_index_order:
        point = Point(current_pose[0], current_pose[1])
        polygon = Polygon([vertex for vertex in all_data[index]['vertice_' + frame]])

        if polygon.contains(point):
            return index

    return None

def get_local_xy_from_latlon(lat, lon, utm_T_local):
    e, n, _, _ = utm.from_latlon(lat, lon)
    utm_pose = np.array([e, n, 0, 1]).T
    local_pose = utm_T_local.dot(utm_pose)
    local_pos = local_pose[:3].T

    return local_pos