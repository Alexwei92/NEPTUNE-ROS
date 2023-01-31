"""
Test VAE Reconstruction Performance
"""
import os
import glob
import numpy as np
import cv2
import pandas
import time

from controller.end_to_end_control import EndToEndController, EndToEndController_TRT

BLACK = (0,0,0)
WHITE = (255,255,255)
RED   = (0,0,255)
BLUE  = (255,0,0)
GREEN = (0,255,0)

def read_data(folder_path, image_only=False):
    # image file
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()

    results = {'color_file_list': color_file_list}

    if not image_only:
        data = pandas.read_csv(os.path.join(folder_path, 'states.csv'))
        # timestamp
        timestamp = data['time'].to_numpy()
        timestamp -= timestamp[0]
        # control cmd
        control_cmd = data['control_cmd'].to_numpy()

        results['timestamp'] = timestamp
        results['control_cmd'] = control_cmd

    return results

def plot_with_cmd(window_name, image_raw, input=0, is_expert=True):
    '''
    Plot image with cmd and return the image
    Input is normalized to [-1, 1]
    '''
    image = image_raw.copy()
    height, width = image.shape[0], image.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    cv2.rectangle(image, (int(w1*width), int(h1*height)), (int(w2*width), int(h2*height)), BLACK, 2) # black
    # plot bar
    bar_width = 5 # pixel
    center_pos = ((w2-w1)*width - bar_width) * (input/2) + 0.5*width
    if is_expert:
        color = RED # red
    else:
        color = BLUE # blue
    cv2.rectangle(image, (int(center_pos-bar_width/2),int(h1*height)), (int(center_pos+bar_width/2),int(h2*height)), color, -1)
    # plot center line
    cv2.line(image, (int(0.5*width),int(h1*height)), (int(0.5*width),int(h2*height)), WHITE, 1) # white
    cv2.imshow(window_name, image) 

    return image


def plot_with_cmd_compare(window_name, image_raw, pilot_input, agent_input):
    '''
    Plot image with cmds and compare
    Input is normalized to [-1, 1]
    '''
    image = image_raw.copy()
    height, width = image.shape[0], image.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    cv2.rectangle(image, (int(w1*width), int(h1*height)), (int(w2*width), int(h2*height)), BLACK, 2) # black
    # plot bar
    bar_width = 5 # pixel
    pilot_center_pos = ((w2-w1)*width - bar_width) * (pilot_input/2) + 0.5*width
    agent_center_pos = ((w2-w1)*width - bar_width) * (agent_input/2) + 0.5*width
    # Pilot input
    cv2.rectangle(image, (int(pilot_center_pos-bar_width/2),int(h1*height)), (int(pilot_center_pos+bar_width/2),int(h2*height)), RED, -1)
    # Agent input
    cv2.rectangle(image, (int(agent_center_pos-bar_width/2),int(h1*height)), (int(agent_center_pos+bar_width/2),int(h2*height)), BLUE, -1)
    # plot center line
    cv2.line(image, (int(0.5*width),int(h1*height)), (int(0.5*width),int(h2*height)), WHITE, 1) # white
    cv2.imshow(window_name, image) 

    return image

if __name__ == '__main__':
    # Datafolder
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/2022-12-15-09-35-34'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-02-56'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-24'
    folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-00'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter2/2023-01-21-09-59-39'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter3/2023-01-24-10-18-32'
    
    # Read data
    data_dict = read_data(folder_path)

    states = pandas.read_csv(os.path.join(folder_path, "states.csv"))
    roll_angle = states['roll_rad'].to_numpy()
    pitch_angle = states['pitch_rad'].to_numpy()
    body_linear_x = states['body_linear_x'].to_numpy()
    body_linear_y = states['body_linear_y'].to_numpy()
    body_angular_z = states['body_angular_z'].to_numpy()
    relative_height = states['odom_rel_height'].to_numpy()

    is_pilot = ~(states['ai_mode'].to_numpy())

    # Load parameter
    model_config = {
        'model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter3/end_to_end/end_to_end_model.pt',
    }

    controller_agent = EndToEndController(**model_config)

    tic = time.time()
    for i in range(len(data_dict['color_file_list'])):
        image_bgr = cv2.imread(data_dict['color_file_list'][i], cv2.IMREAD_UNCHANGED)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        # state extra
        state_extra = np.array([
            roll_angle[i],
            pitch_angle[i],
            body_linear_x[i],
            body_linear_y[i],
            body_angular_z[i],
            relative_height[i],
        ])

        # predict
        predicted_action = controller_agent.predict(image_bgr, is_bgr=True, state_extra=state_extra)

        if abs(predicted_action) < 1e-2:
            predicted_action = 0.0

        # plot
        plot_with_cmd_compare('control', image_bgr, data_dict['control_cmd'][i], predicted_action)

        elapsed_time = time.time() - tic
        # time.sleep(max(data_dict['timestamp'][i] - elapsed_time, 0))
