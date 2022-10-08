import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import transformation

class FieldMapPlot():
    def __init__(self,
        data,
        utm_T_local,
        field_bound,
        frame='local',
    ):
        self.row_data = data['row']
        self.column_data = data['column']
        self.utm_T_local = utm_T_local
        self.field_bound_latlon = field_bound['latlon']
        self.field_bound_local = field_bound['local']

        if frame not in {'latlon', 'local'}:
            frame = 'local'
        self.frame = frame
        print("Display in %s frame" % self.frame)
        
        self.fig, self.axis = plt.subplots()
        self.initialize_map()

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

    def draw_vertice(self, vertice, s=10, color='b'):
        self.axis.scatter(vertice[:,0], vertice[:,1], s=s, color=color)

    def draw_line(self, line, color='k', linestyle='-', linewidth=0.5):
        self.axis.plot(line[:,0], line[:,1], color=color, linestyle=linestyle, linewidth=linewidth)

    def draw_polygon(self, vertice, color='b', linestyle='-', linewidth=3.0, alpha=0.05):
        self.axis.add_patch(patches.Polygon(vertice, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha))

    def draw_data(self, data, vertice=True, treelines=True, centerline=True, polygon=True):
        # vertice
        if vertice:
            self.draw_vertice(data['vertice_' + self.frame])
        # treeline
        if treelines:
            self.draw_line(data['treelines_' + self.frame][0])
            self.draw_line(data['treelines_' + self.frame][1])
        # centerline
        if centerline:
            self.draw_line(data['centerline_' + self.frame], color='r', linestyle='--')
        # polygon
        if polygon:
            self.draw_polygon(data['vertice_' + self.frame])

    def draw_all_rows(self):
        for i in range(len(self.row_data)):
            self.draw_data(self.row_data[i])

    def draw_all_columns(self):
        for i in range(len(self.column_data)):
            self.draw_data(self.column_data[i])
