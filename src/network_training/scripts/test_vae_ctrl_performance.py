"""
Test VAE Reconstruction Performance
"""
import os
import glob
import numpy as np
import cv2
import pandas
import time
import argparse

from controller.vae_latent_control import VAELatentController_Full, VAELatentController, VAELatentController_TRT

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
        # ai mode
        ai_mode = data['ai_mode'].to_numpy()

        results['timestamp'] = timestamp
        results['control_cmd'] = control_cmd
        results['ai_mode'] = ai_mode

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


def plot_with_cmd_compare(window_name, image_raw, pilot_input, agent_input, is_pilot_human=True):
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
    if is_pilot_human:
        c = RED
    else:
        c = GREEN
    cv2.rectangle(image, (int(pilot_center_pos-bar_width/2),int(h1*height)), (int(pilot_center_pos+bar_width/2),int(h2*height)), c, -1)
    # Agent input
    cv2.rectangle(image, (int(agent_center_pos-bar_width/2),int(h1*height)), (int(agent_center_pos+bar_width/2),int(h2*height)), BLUE, -1)
    # plot center line
    cv2.line(image, (int(0.5*width),int(h1*height)), (int(0.5*width),int(h2*height)), WHITE, 1) # white
    cv2.imshow(window_name, image) 

    return image

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-extra", "--extra", default=False, action="store_true", help="enable state extra")
    
    args = argParser.parse_args()
    enable_extra = args.extra

    # Datafolder
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/2022-12-15-09-35-34'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-02-56'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-24'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-00'
    folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter2/2023-01-21-09-59-39'
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
    if enable_extra:
        model_config = {
            'vae_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/vanilla_vae/vanilla_vae_model_z_1000.pt',
            'latent_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter3/latent_ctrl_with_extra/latent_ctrl_vanilla_vae_model_z_1000.pt',
            'model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter3/combined_vae_latent_ctrl_z_1000_with_extra.pt',
            # 'tensorrt_engine_path': '/home/lab/catkin_ws/src/neptune-ros/model_weight/vae/combined_vae_latent_ctrl_z_1000.trt',
            'enable_extra': True,
        }
    else:
        model_config = {
            'vae_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/vanilla_vae/vanilla_vae_model_z_1000.pt',
            'latent_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter3/latent_ctrl_no_extra/latent_ctrl_vanilla_vae_model_z_1000.pt',
            'model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter3/combined_vae_latent_ctrl_z_1000_no_extra.pt',
            # 'tensorrt_engine_path': '/home/lab/catkin_ws/src/neptune-ros/model_weight/vae/combined_vae_latent_ctrl_z_1000.trt',
            'enable_extra': False,
        }

    print(f"enable_extra is {model_config['enable_extra']}")
    controller_agent = VAELatentController(**model_config)
    controller_agent_full = VAELatentController_Full(**model_config)
    # controller_agent_trt = VAELatentController_TRT(**model_config)

    # vae_out = cv2.VideoWriter('vae_output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (640, 240))
    # control_out = cv2.VideoWriter('control_output.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (640, 480))

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
        predicted_action = controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra)
        # predicted_action1 = controller_agent.predict(image_bgr, is_bgr=True, state_extra=state_extra)
        # predicted_action2 = controller_agent_trt.predict(image_bgr, is_bgr=True, state_extra=state_extra)
        image_raw, image_pred = controller_agent_full.reconstruct_image(image_bgr, is_bgr=True)

        if abs(predicted_action) < 1e-2:
            predicted_action = 0.0

        # if abs(predicted_action1) < 1e-2:
        #     predicted_action1 = 0.0

        # if abs(predicted_action2) < 1e-2:
        #     predicted_action2 = 0.0

        # print(predicted_action, predicted_action1, predicted_action2)

        # plot
        image_raw = cv2.resize(image_raw, (320, 240))
        image_pred = cv2.resize(image_pred, (320, 240))
        image_combine = np.concatenate((image_raw, image_pred), axis=1)
        cv2.imshow('VAE', cv2.cvtColor(image_combine, cv2.COLOR_RGB2BGR))
        # vae_out.write(cv2.cvtColor(image_combine, cv2.COLOR_RGB2BGR))

        plot_with_cmd_compare('control', image_bgr, data_dict['control_cmd'][i], predicted_action, ~data_dict['ai_mode'][i])
        # cv_image = plot_with_cmd('control', image_bgr, data_dict['control_cmd'][i], is_expert=is_pilot[i])
        # control_out.write(cv_image)

        elapsed_time = time.time() - tic
        # time.sleep(max(data_dict['timestamp'][i] - elapsed_time, 0))
    
    # vae_out.release()
    # control_out.release()