import numpy as np
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.navigation_utils import find_area_index

def plot_vehicle(handle, pos, heading, show_FOV=True, is_first=False):
    """
    plot the vehicle with FOV
    """
    FOV = 70 * np.pi / 180
    # location
    if is_first:
        origin, = handle.plot(pos[0], pos[1], marker='o', markersize=3, fillstyle='full', color='r')
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
            FOV_patch = handle.add_patch(patches.Polygon(path, color='g', linestyle='none', alpha=0.25))
        else:
            handle['FOV_patch'].set_xy(path)

    if is_first:
        return {'origin': origin,
                'forward_line': forward_line,
                'right_line': right_line, 
                'FOV_patch': FOV_patch}  

def plot_trajectory_history(handle, trajectory_history, is_first=False):
    """
    plot trajectory history
    """
    pos_x, pos_y = [], []

    for i in range(len(trajectory_history)):
        pos_x.append(trajectory_history[i][0])
        pos_y.append(trajectory_history[i][1])    

    if is_first:
        trajectory, = handle.plot(pos_x, pos_y, color='c', alpha=0.7, linewidth=1.0)
        return trajectory
    else:
        handle['trajectory_history'].set_xdata(pos_x)
        handle['trajectory_history'].set_ydata(pos_y)


class FieldMapPlot():
    def __init__(self,
        data,
        field_bound,
        frame='local',
    ):
        self.data = data
        self.field_bound_latlon = field_bound['latlon']
        self.field_bound_local = field_bound['local']

        if frame not in {'latlon', 'local'}:
            frame = 'local'
        self.frame = frame
        
        self.fig, self.axis = plt.subplots()
        self.initialize_variables()
        self.initialize_map()

    def initialize_variables(self):
        self.has_initialized = False
        self.current_area = None
        self.pose_history = deque(maxlen=100)
        self.last_index = None

    def initialize_map(self):
        if self.frame == 'latlon':
            self.axis.set_xlabel('Longitude [deg]')
            self.axis.set_ylabel('Latitude [deg]')
            self.axis.set_aspect(1)
            self.axis.set_xlim(self.field_bound_latlon[0])
            self.axis.set_ylim(self.field_bound_latlon[1])
            # self.axis.set_xticks(np.arange(self.field_bound_latlon[0][0], self.field_bound_latlon[0][1], 0.0003))
            # self.axis.set_yticks(np.arange(self.field_bound_latlon[1][0], self.field_bound_latlon[1][1], 0.0003))
            self.axis.ticklabel_format(useOffset=False)

        if self.frame == 'local':
            self.axis.set_xlabel('x [m]')
            self.axis.set_ylabel('y [m]')
            self.axis.set_aspect(1)
            self.axis.set_xlim(self.field_bound_local[0])
            self.axis.set_ylim(self.field_bound_local[1])
            # self.axis.set_xticks(np.arange(np.floor(self.field_bound_local[0][0]), np.ceil(self.field_bound_local[0][1]), 20))
            # self.axis.set_yticks(np.arange(np.floor(self.field_bound_local[1][0]), np.ceil(self.field_bound_local[1][1]), 20))
            self.axis.ticklabel_format(useOffset=False)

        self.draw_all_data()
        self.fig.tight_layout()

    def draw_vertice(self, vertice, s=10, color='b'):
        self.axis.scatter(vertice[:,0], vertice[:,1], s=s, color=color)

    def draw_line(self, line, color='k', linestyle='-', linewidth=0.5):
        self.axis.plot(line[:,0], line[:,1], color=color, linestyle=linestyle, linewidth=linewidth)

    def draw_polygon(self, vertice, color='b', linestyle='-', linewidth=3.0, alpha=0.1):
        self.axis.add_patch(patches.Polygon(vertice, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha))

    def draw_single_data(self, data, vertice=True, treelines=True, centerline=True, polygon=False):
        # vertice
        if vertice:
            self.draw_vertice(data['vertice_' + self.frame])
        # treeline
        if treelines:
            self.draw_line(data['treelines_' + self.frame][0])
            self.draw_line(data['treelines_' + self.frame][1])
        # centerline
        if centerline:
            self.draw_line(data['centerline_' + self.frame], color='r', linestyle='-.')
        # polygon
        if polygon:
            self.draw_polygon(data['vertice_' + self.frame])

    def draw_all_data(self):
        for i in range(len(self.data)):
            self.draw_single_data(self.data[i])

    def update_graph(self, pos, heading):
        self.pose_history.append([pos[0], pos[1], heading])
        if self.has_initialized:
            plot_vehicle(self.axis_dict, pos, heading, is_first=False)
            plot_trajectory_history(self.axis_dict, self.pose_history)
        else:
            self.axis_dict = plot_vehicle(self.axis, pos, heading, is_first=True)
            self.axis_dict['trajectory_history'] = plot_trajectory_history(self.axis, self.pose_history, is_first=True)
            self.has_initialized = True

        index = find_area_index(
            [pos[0], pos[1]],
            self.data,
            frame=self.frame,
        )

        if index != self.last_index: 
            if index is None:
                if self.current_area is not None:
                    self.current_area.set_xy(np.empty((0,2)))
                    self.current_area = None
            else:
                if self.current_area is None:
                    self.current_area = self.axis.add_patch(patches.Polygon(
                        self.data[index]['vertice_' + self.frame], 
                        color='C0', linestyle='-', linewidth=3.0, alpha=0.2
                    ))
                else:
                    self.current_area.set_xy(self.data[index]['vertice_' + self.frame])
        self.last_index = index