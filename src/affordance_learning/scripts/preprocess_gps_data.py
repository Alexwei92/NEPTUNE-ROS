import os
import pandas
import utm
import numpy as np
import pickle

from utils import transformation

curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "../../../data/field_map/")
save_dir = os.path.join(curr_dir, 'ground_truth')

output_filename = 'plant_field.pkl'

if __name__ == "__main__":
    # load data
    south_data = "2022_10_04_09_57_34/gps_at_trigger.csv"
    west_data = "2022_10_04_10_48_09/gps_at_trigger.csv"
    north_path = "2022_10_04_11_05_46/gps_at_trigger.csv"
    east_path = "2022_10_04_11_41_05/gps_at_trigger.csv"

    all_data_csv = [south_data, west_data, north_path, east_path]
    all_data_name = ['south', 'west', 'north', 'east']

    # raw pandas data
    all_data_pt_raw = {}
    for data_csv, name in zip(all_data_csv, all_data_name) :
        raw_data = pandas.read_csv(os.path.join(data_dir, data_csv))
        all_data_pt_raw[name] = (raw_data)

    # field bound latlon
    field_bound_latlon = (
        (-121.794980, -121.793920), # longitude
        (38.537990, 38.539540),     # latitude
    )

    # field origin lat, lon
    field_origin = (field_bound_latlon[1][0], field_bound_latlon[0][0])
    e, n, zone_number, zone_letter = utm.from_latlon(field_origin[0], field_origin[1]) 
    # print('easting: ' + str(e) + ', northing: ' + str(n))
    # print('zone number: {zone_number}, zone_letter: {zone_letter}')

    # get transformation
    T = transformation.states2SE3([e, n, 0, 0, 0, 0])
    R = transformation.states2SE3([0, 0, 0, 0, 0, 0])
    local_T_utm = R.dot(T)
    utm_T_local = np.linalg.inv(local_T_utm)
    # print('utm_T_local: \n' + str(utm_T_local))

    # field bound local
    field_bound_local = [
        [0, 0],
        [0, 0],
    ]
    for i in range(2):
        for j in range(2):
            e, n, _, _ = utm.from_latlon(field_bound_latlon[1][i], field_bound_latlon[0][j])
            point_local = utm_T_local.dot(np.array([e, n, 0, 1]).T)
            field_bound_local[0][0] = min(field_bound_local[0][0], point_local[0])
            field_bound_local[0][1] = max(field_bound_local[0][1], point_local[0])
            field_bound_local[1][0] = min(field_bound_local[1][0], point_local[1])
            field_bound_local[1][1] = max(field_bound_local[1][1], point_local[1])

    field_bound_local = (
        (field_bound_local[0][0], field_bound_local[0][1]),
        (field_bound_local[1][0], field_bound_local[1][1]),
    )

    #####################################
    # convert to numpy
    all_data_latlon = {}
    for key in all_data_pt_raw.keys():
        all_data_latlon[key] = np.vstack((all_data_pt_raw[key].lon_avg, all_data_pt_raw[key].lat_avg)).T

    # convert to utm
    all_data_local = {}
    for key in all_data_pt_raw.keys():
        easting, northing, _, _ = utm.from_latlon(all_data_latlon[key][:, 1], all_data_latlon[key][:, 0])
        utm_full = np.vstack((easting, northing, np.zeros_like(easting), np.ones_like(easting)))
        local_full = utm_T_local.dot(utm_full)
        all_data_local[key] = local_full.T[:, :2]

    # convert to treeline
    treeline_WE_local = []
    for i in range(len(all_data_local['west'])):
        treeline_WE_local.append(np.vstack((
            all_data_local['west'][i],
            all_data_local['east'][i],
        )))

    treeline_SN_local = []
    for i in range(len(all_data_local['south'])):
        treeline_SN_local.append(np.vstack((
            all_data_local['south'][i],
            all_data_local['north'][i],
        )))

    # remove offset
    ft2m = 0.3048
    west_offset = 15 * ft2m
    east_offset = 10 * ft2m
    south_offset = 15 * ft2m
    north_offset = 15 * ft2m

    treeline_WE_local_actual = []
    for i in range(len(all_data_latlon['west'])):
        # local
        treeline = [
            all_data_local['west'][i],
            all_data_local['east'][i],
        ]
        treeline_length = np.linalg.norm(treeline[1] - treeline[0])
        alpha_west = west_offset / treeline_length
        alpha_east = east_offset / treeline_length

        new_pt_west = (treeline[1] - treeline[0]) * alpha_west + treeline[0]
        new_pt_east = (treeline[1] - treeline[0]) * (1 - alpha_east) + treeline[0]

        treeline_WE_local_actual.append(np.vstack((
            new_pt_west,
            new_pt_east,
        )))

    treeline_SN_local_actual = []
    for i in range(len(all_data_latlon['south'])):
        treeline = [
            all_data_local['south'][i],
            all_data_local['north'][i],
        ]
        treeline_length = np.linalg.norm(treeline[1] - treeline[0])
        alpha_south = south_offset / treeline_length
        alpha_north = north_offset / treeline_length

        new_pt_south = (treeline[1] - treeline[0]) * alpha_south + treeline[0]
        new_pt_north = (treeline[1] - treeline[0]) * (1 - alpha_north) + treeline[0]

        treeline_SN_local_actual.append(np.vstack((
            new_pt_south,
            new_pt_north,
        )))

    # column-wise (index 0 start from west)
    column_data = []
    for i in range(len(treeline_SN_local)-1):
        column_data.append({
            'index': i,
            'vertice': np.vstack((
                all_data_local['south'][i],
                all_data_local['north'][i],
                all_data_local['north'][i+1],
                all_data_local['south'][i+1],
            )),
            'vertice_actual': np.vstack((
                treeline_SN_local_actual[i][0],
                treeline_SN_local_actual[i][1],
                treeline_SN_local_actual[i+1][1],
                treeline_SN_local_actual[i+1][0],
            )),
            'treelines': [
                treeline_SN_local[i],
                treeline_SN_local[i+1],
            ],
            'treelines_actual': [
                treeline_SN_local_actual[i],
                treeline_SN_local_actual[i+1],
            ],
            'centerline': (treeline_SN_local[i] + treeline_SN_local[i+1]) / 2.0,
            'centerline_actual': (treeline_SN_local_actual[i] + treeline_SN_local_actual[i+1]) / 2.0,
        })

    # row_wise (index 0 start from south)
    row_data = []
    for i in range(len(treeline_WE_local)-1):
        row_data.append({
            'index': i,
            'vertice': np.vstack((
                all_data_local['west'][i],
                all_data_local['east'][i],
                all_data_local['east'][i+1],
                all_data_local['west'][i+1],
            )),
            'vertice_actual': np.vstack((
                treeline_WE_local_actual[i][0],
                treeline_WE_local_actual[i][1],
                treeline_WE_local_actual[i+1][1],
                treeline_WE_local_actual[i+1][0],
            )),
            'treelines': [
                treeline_WE_local[i],
                treeline_WE_local[i+1],
            ],
            'treelines_actual': [
                treeline_WE_local_actual[i],
                treeline_WE_local_actual[i+1],
            ],
            'centerline': (treeline_WE_local[i] + treeline_WE_local[i+1]) / 2.0,
            'centerline_actual': (treeline_WE_local_actual[i] + treeline_WE_local_actual[i+1]) / 2.0,
        })

    # save output
    file_path = os.path.join(save_dir, output_filename)
    output_results = {
        'row_data': row_data,
        'column_data': column_data,
        'utm_T_local': utm_T_local,
        'field_bound_latlon': field_bound_latlon,
        'field_bound_local': field_bound_local,      
    }

    with open(file_path, "wb") as f:
        pickle.dump(output_results, f, protocol=2)
        print("Data saved to: " + file_path)
