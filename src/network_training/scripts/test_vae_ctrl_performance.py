"""
Test VAE Reconstruction Performance
"""
import os
import glob
import numpy as np
import cv2
import pandas
import time

from controller.vae_latent_control import VAECtrl, VAELatentController

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
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06'
    folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/2022-12-15-09-35-34'

    # Read data
    data_dict = read_data(folder_path)

    states = pandas.read_csv(os.path.join(folder_path, "states.csv"))
    roll_angle = states['roll_rad'].to_numpy()
    pitch_angle = states['pitch_rad'].to_numpy()
    body_linear_x = states['body_linear_x'].to_numpy()
    body_linear_y = states['body_linear_y'].to_numpy()
    body_angular_z = states['body_angular_z'].to_numpy()
    state_extra = np.array([body_linear_x, body_linear_y, body_angular_z]).T

    # Load VAE parameter
    model_config = {
        'vae_model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/vanilla_vae/vanilla_vae_model_z_1000.pt',
        'latent_model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/latent_ctrl/latent_ctrl_vanilla_vae_model_z_1000.pt',
        'model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/combined_vae_latent_ctrl_z_1000.pt'
    }

    controller_agent = VAECtrl(**model_config)
    test_controller_agent = VAELatentController(**model_config)

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
        ])

        # predict
        predicted_action = controller_agent.predict(image_bgr, is_bgr=True, state_extra=state_extra)
        test_predicted_action = test_controller_agent.predict(image_bgr, is_bgr=True, state_extra=state_extra)
        image_raw, image_pred = controller_agent.reconstruct_image(image_bgr, is_bgr=True)

        if abs(predicted_action) < 1e-2:
            predicted_action = 0.0

        if abs(test_predicted_action) < 1e-2:
            test_predicted_action = 0.0

        # print(predicted_action, test_predicted_action)

        # plot
        image_raw = cv2.resize(image_raw, (320, 240))
        image_pred = cv2.resize(image_pred, (320, 240))
        image_combine = np.concatenate((image_raw, image_pred), axis=1)
        cv2.imshow('VAE', cv2.cvtColor(image_combine, cv2.COLOR_RGB2BGR))

        plot_with_cmd_compare('control', image_bgr, data_dict['control_cmd'][i], predicted_action)

        elapsed_time = time.time() - tic
        time.sleep(max(data_dict['timestamp'][i] - elapsed_time, 0))