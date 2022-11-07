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
    if is_first:
        # x_range, y_range = handle.get_xlim(), handle.get_ylim()
        # L = min((x_range[1]-x_range[0]) / 30, (y_range[1]-y_range[0]) / 30)
        L = 0.5
    else:
        L = handle['L']
    # location
    if is_first:
        origin, = handle.plot(pos[0], pos[1], marker='o', markersize=3, fillstyle='full', color='r')
    else:
        handle['origin'].set_xdata(pos[0])
        handle['origin'].set_ydata(pos[1])
    # heading
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
        L_fov = 2.5*L / np.cos(FOV/2)
        left_point = [pos[0] + L_fov*np.cos(heading+FOV/2), pos[1] + L_fov*np.sin(heading+FOV/2)]
        right_point = [pos[0] + L_fov*np.cos(heading-FOV/2), pos[1] + L_fov*np.sin(heading-FOV/2)]
        path = [pos, left_point, right_point]
        if is_first:
            FOV_patch = handle.add_patch(patches.Polygon(path, color='g', linestyle='none', alpha=0.2))
        else:
            handle['FOV_patch'].set_xy(path)

    if is_first:
        return {'origin': origin,
                'forward_line': forward_line,
                'right_line': right_line, 
                'FOV_patch': FOV_patch,
                'L': L}  


def plot_trajectory_history(handle, trajectory_history, is_first=False, name='trajectory_history', color='c', linestyle='-', linewidth=1.5, alpha=0.7):
    """
    plot trajectory history
    """
    pos_x, pos_y = [], []

    for i in range(len(trajectory_history)):
        pos_x.append(trajectory_history[i][0])
        pos_y.append(trajectory_history[i][1])    

    if is_first:
        trajectory, = handle.plot(pos_x, pos_y, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        return trajectory
    else:
        handle[name].set_xdata(pos_x)
        handle[name].set_ydata(pos_y)


def plot_current_area(handle, data, is_first=False):
    if is_first:
        current_area_dict = {}
        current_area_dict['treeline1'], = handle.plot(
            data['treelines_actual'][0][:, 0], data['treelines_actual'][0][:, 1],
            color='g', linestyle='-', linewidth=2.0,
        )
        current_area_dict['treeline2'], = handle.plot(
            data['treelines_actual'][1][:, 0], data['treelines_actual'][1][:, 1],
            color='g', linestyle='-', linewidth=2.0,
        )
        current_area_dict['centerline'], = handle.plot(
            data['centerline'][:, 0], data['centerline'][:, 1],
            color='k', linestyle='-.', linewidth=1.0,
        )
        current_area_dict['polygon'] = handle.add_patch(patches.Polygon(
            data['vertice_actual'], 
            color='C0', linestyle='-', linewidth=3.0, alpha=0.2
        ))

        return current_area_dict

    else:
        handle['treeline1'].set_xdata(data['treelines_actual'][0][:, 0])
        handle['treeline1'].set_ydata(data['treelines_actual'][0][:, 1])
        handle['treeline2'].set_xdata(data['treelines_actual'][1][:, 0])
        handle['treeline2'].set_ydata(data['treelines_actual'][1][:, 1])
        handle['centerline'].set_xdata(data['centerline'][:, 0])
        handle['centerline'].set_ydata(data['centerline'][:, 1])
        handle['polygon'].set_xy(data['vertice_actual'])


class FieldMapPlot():
    def __init__(self,
        data,
        field_bound,
    ):
        self.data = data
        self.field_bound_latlon = field_bound['latlon']
        self.field_bound_local = field_bound['local']

        self.frame = 'local'
        
        self.fig = plt.figure()
        gs = self.fig.add_gridspec(2,2)
        self.axis_full = self.fig.add_subplot(gs[:, 0])
        self.axis_zoomin = self.fig.add_subplot(gs[0, 1])
        self.axis_camera = self.fig.add_subplot(gs[1, 1])

        self.initialize_variables()
        self.initialize_map()

    def initialize_variables(self):
        self.has_initialized = False
        self.pose_history = deque(maxlen=100) # onboard GPS
        self.pose2_history = deque(maxlen=100) # piksi
        self.current_area = None
        self.last_index = None
        self.last_image = None

    def initialize_map(self):
        # full
        self.axis_full.set_xlabel('x [m]')
        self.axis_full.set_ylabel('y [m]')
        self.axis_full.set_aspect(1)
        self.axis_full.set_xlim(self.field_bound_local[0])
        self.axis_full.set_ylim(self.field_bound_local[1])
        # self.axis_full.set_xticks(np.arange(np.floor(self.field_bound_local[0][0]), np.ceil(self.field_bound_local[0][1]), 20))
        # self.axis_full.set_yticks(np.arange(np.floor(self.field_bound_local[1][0]), np.ceil(self.field_bound_local[1][1]), 20))
        self.axis_full.ticklabel_format(useOffset=False)
        # self.axis_full.tick_params(direction='out', length=0, color='k')

        # zoom in
        self.axis_zoomin.set_aspect(1)
        # self.axis_zoomin.set_xticklabels([])
        # self.axis_zoomin.set_yticklabels([])
        # self.axis_zoomin.tick_params(direction='out', length=0, color='w')

        # camera
        self.axis_camera.set_aspect(1)
        self.axis_camera.set_xticklabels([])
        self.axis_camera.set_yticklabels([])
        self.axis_camera.tick_params(direction='out', length=0, color='w')

        self.draw_all_data(self.axis_full)
        # self.draw_all_data(self.axis_zoomin)
        self.fig.tight_layout()

    def draw_vertice(self, axis, vertice, s=10, color='b'):
        axis.scatter(vertice[:,0], vertice[:,1], s=s, color=color)

    def draw_line(self, axis, line, color='k', linestyle='-', linewidth=1.0):
        axis.plot(line[:,0], line[:,1], color=color, linestyle=linestyle, linewidth=linewidth)

    def draw_polygon(self, axis, vertice, color='b', linestyle='-', linewidth=3.0, alpha=0.1):
        axis.add_patch(patches.Polygon(vertice, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha))

    def draw_single_data(self, axis, data, vertice=True, treelines=True, centerline=True, polygon=False):
        # vertice
        if vertice:
            self.draw_vertice(axis, data['vertice'])
        # treeline
        if treelines:
            self.draw_line(axis, data['treelines'][0])
            self.draw_line(axis, data['treelines'][1])
        # centerline
        if centerline:
            self.draw_line(axis, data['centerline'], color='r', linestyle='-.', linewidth=0.5)
        # polygon
        if polygon:
            self.draw_polygon(axis, data['vertice'])

    def draw_all_data(self, axis):
        for i in range(len(self.data)):
            self.draw_single_data(axis, self.data[i])

    def update_graph(self, pos, heading, num_sat=0, fix_type='No GPS', 
                    pos2=None, num_sat2=None, fix_type2=None):
        self.pose_history.append([pos[0], pos[1], heading])
        if pos2 is not None:
            self.pose2_history.append([pos2[0], pos2[1], heading])
            disp_second_gps = True
        else:
            disp_second_gps = False

        if self.has_initialized:
            # full
            plot_vehicle(self.axis_dict, pos, heading, is_first=False)
            plot_trajectory_history(self.axis_dict, self.pose_history)
            # zoom in
            plot_vehicle(self.axis_dict_zoom, pos, heading, is_first=False)
            plot_trajectory_history(self.axis_dict_zoom, self.pose_history)
            if disp_second_gps:
                plot_trajectory_history(self.axis_dict_zoom, self.pose2_history, name='trajectory2_history')
            self.update_zoom_in(self.axis_dict_zoom)
        else:
            # full
            self.axis_dict = plot_vehicle(self.axis_full, pos, heading, is_first=True)
            self.axis_dict['trajectory_history'] = plot_trajectory_history(self.axis_full, self.pose_history, is_first=True)
            # zoom in
            self.axis_dict_zoom = plot_vehicle(self.axis_zoomin, pos, heading, is_first=True)
            self.axis_dict_zoom['trajectory_history'] = plot_trajectory_history(self.axis_zoomin, self.pose_history, is_first=True, color='C1')
            if disp_second_gps:
                self.axis_dict_zoom['trajectory2_history'] = plot_trajectory_history(self.axis_zoomin, self.pose2_history, is_first=True, color='C0')
            self.update_zoom_in(self.axis_dict_zoom)
            self.has_initialized = True

        # gps status
        if disp_second_gps:
            title_name = ("GPS Count: %d, GPS Type: %s \n GPS2 Count: %d, GPS2 Type: %s" % (num_sat, fix_type, num_sat2, fix_type2))
        else:
            title_name = ("GPS Count: %d, GPS Type: %s" % (num_sat, fix_type))
        self.axis_full.set_title(
            title_name,
            {'fontsize': 10, 'fontweight': 'normal'}, pad=1.5,
        )

        # find index
        index = find_area_index(
            [pos[0], pos[1]],
            self.data,
            start_index = self.last_index if self.last_index is not None else 0,
        )

        if index != self.last_index: 
            if index is None:
                if self.current_area is not None:
                    # full
                    self.current_area.set_xy(np.empty((0,2)))
                    self.current_area = None
                    # zoom in
                    self.current_area_zoom['treeline1'].set_xdata([])
                    self.current_area_zoom['treeline1'].set_ydata([])
                    self.current_area_zoom['treeline2'].set_xdata([])
                    self.current_area_zoom['treeline2'].set_ydata([])
                    self.current_area_zoom['centerline'].set_xdata([])
                    self.current_area_zoom['centerline'].set_ydata([])
                    self.current_area_zoom['polygon'].set_xy(np.empty((0,2)))
                    self.current_area_zoom = None
            else:
                if self.current_area is None:
                    # full
                    self.current_area = self.axis_full.add_patch(patches.Polygon(
                        self.data[index]['vertice_actual'], 
                        color='C0', linestyle='-', linewidth=3.0, alpha=0.2
                    ))
                    # zoom in
                    self.current_area_zoom = plot_current_area(self.axis_zoomin, self.data[index], is_first=True)
                else:
                    # full
                    self.current_area.set_xy(self.data[index]['vertice_actual'])
                    # zoom in
                    plot_current_area(self.current_area_zoom, self.data[index])
        
        self.last_index = index

        if index is not None:
            self.axis_zoomin.set_title(
                "Index: %d" % index,
                {'fontsize': 10, 'fontweight': 'normal', 'color': 'g'}, pad=2.0,
            )
        else:
            self.axis_zoomin.set_title(                
                "Idle", 
                {'fontsize': 10, 'fontweight': 'normal', 'color': 'r'}, pad=2.0,
            )

    def update_zoom_in(self, axis_dict):
        pos_x = axis_dict['origin'].get_xdata()
        pos_y = axis_dict['origin'].get_ydata()
        L = axis_dict['L']
        dist = 5
        x_range = [pos_x - dist, pos_x + dist]
        y_range = [pos_y - dist, pos_y + dist]
        self.axis_zoomin.set_xlim(x_range)
        self.axis_zoomin.set_ylim(y_range)

    def update_image(self, image):
        if image is not None:
            img = image.copy()
        else:
            img = np.zeros((100,100,3))

        if self.last_image is None:
            self.last_image = self.axis_camera.imshow(img)
        else:
            self.last_image.set_data(img)
        

def plot_single_data(axis, data):
    # vertice
    axis.scatter(data['vertice'][:,0], data['vertice'][:,1], s=10, color='b')
    # treeline
    axis.plot(data['treelines_actual'][0][:,0], data['treelines_actual'][0][:,1], color='g', linestyle='-', linewidth=2.0)
    axis.plot(data['treelines_actual'][1][:,0], data['treelines_actual'][1][:,1], color='g', linestyle='-', linewidth=2.0)
    # centerline
    axis.plot(data['centerline_actual'][:,0], data['centerline_actual'][:,1], color='k', linestyle='-.', linewidth=0.5)
    # polygon
    axis.add_patch(patches.Polygon(data['vertice_actual'], color='C0', linestyle='-', linewidth=3.0, alpha=0.1))

    axis.margins(y=0.5)
    axis.set_aspect(1.0)
    # axis.set_xlabel('x [m]')
    # axis.set_ylabel('y [m]')