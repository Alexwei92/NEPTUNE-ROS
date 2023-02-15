"""
Plot Histogram of Command
"""
import os
import glob
import pandas
import cv2
import numpy as np
import matplotlib.pyplot as plt

from controller.vae_latent_control import VAELatentController

def read_data(folder_path):
    # image file
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()

    mavros_data = pandas.read_csv(os.path.join(folder_path, 'states.csv'))
    # ai_mode
    ai_mode = mavros_data['ai_mode'].to_numpy()
    is_pilot = (~ai_mode)

    # state extra
    roll = mavros_data['roll_rad'].to_numpy(dtype=np.float32)[is_pilot]
    pitch = mavros_data['pitch_rad'].to_numpy(dtype=np.float32)[is_pilot]
    linear_x = mavros_data['body_linear_x'].to_numpy()[is_pilot]
    linear_y = mavros_data['body_linear_y'].to_numpy()[is_pilot]
    angular_z = mavros_data['body_angular_z'].to_numpy()[is_pilot]
    relative_height = mavros_data['odom_rel_height'].to_numpy()[is_pilot]
    
    state_extra = np.array([
        roll,
        pitch,
        linear_x,
        linear_y,
        angular_z,
        relative_height,
    ]).T
   
    # control cmd
    control_cmd = mavros_data['control_cmd'].to_numpy()[is_pilot]
    control_cmd[abs(control_cmd) < 1e-2] = 0.0
    
    color_file_list_pilot = []
    for flag, color_file in zip(is_pilot, color_file_list):
        if flag == True:
            color_file_list_pilot.append(color_file)

    results = {}
    results['color_file_list'] = color_file_list_pilot
    results['control_cmd'] = control_cmd
    results['state_extra'] = state_extra

    return results


if __name__ == "__main__":
    # Datafolder
    dataset_dir = '/media/lab/NEPTUNE2/field_datasets/human_data'

    # Iteration
    max_iter = 1

    # Load parameter
    model_config = {
        'model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter1/combined_vae_latent_ctrl_z_1000.pt',
        'enable_extra': True,
    }
    controller_agent = VAELatentController(**model_config)

    ######
    all_pilot_cmd = np.empty((0,))
    all_agent_cmd = np.empty((0,))
    all_state_extra = np.empty((0, 6))
    for iteration in range(max_iter+1):
        folder_path = os.path.join(dataset_dir, 'iter' + str(iteration))
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            print(subfolder_path)
            data_dict = read_data(subfolder_path)
            all_pilot_cmd = np.concatenate((all_pilot_cmd, data_dict['control_cmd']), axis=0)
            all_state_extra = np.concatenate((all_state_extra, data_dict['state_extra']), axis=0)

            for i in range(len(data_dict['color_file_list'])):
                image_bgr = cv2.imread(data_dict['color_file_list'][i], cv2.IMREAD_UNCHANGED)
                state_extra = data_dict['state_extra'][i, :]
                agent_cmd = controller_agent.predict(image_bgr, is_bgr=True, state_extra=state_extra)
                if abs(agent_cmd) < 1e-2:
                    agent_cmd = 0.0
                all_agent_cmd = np.concatenate((all_agent_cmd,  np.array([agent_cmd])), axis=0)
    
    N_zero = sum(all_pilot_cmd==0)
    zero_index = []
    for idx, value in enumerate(all_pilot_cmd):
        if value == 0:
            zero_index.append(idx)

        if len(zero_index) > N_zero * 0.85:
            break
    all_pilot_cmd = np.delete(all_pilot_cmd, zero_index)

    N_zero = sum(all_agent_cmd==0)
    zero_index = []
    for idx, value in enumerate(all_agent_cmd):
        if value == 0:
            zero_index.append(idx)

        if len(zero_index) > N_zero * 0.85:
            break
    all_agent_cmd = np.delete(all_agent_cmd, zero_index)

    # Plot the Command Histogram
    plt.figure()
    plt.xlim([-1.1, 1.1])
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.xlabel('Command')
    plt.ylabel('%')
    plt.hist(all_pilot_cmd, bins=50, range=(-1,1), density=True, color='b', alpha=0.75, label='Pilot')
    plt.hist(all_agent_cmd, bins=50, range=(-1,1), density=True, color='r', alpha=0.75, label='AI Predicted')
    plt.legend()

    # Plot the State Extra Histogram
    fig, axis = plt.subplots(3,2)

    axis[0][0].hist(all_state_extra[:,0], bins=50, density=True, color='b', alpha=0.75)
    axis[0][0].set_title('roll_angle')
    axis[0][0].set_xlabel('[rad]')

    axis[1][0].hist(all_state_extra[:,1], bins=50, density=True, color='b', alpha=0.75)
    axis[1][0].set_title('pitch_angle')
    axis[1][0].set_xlabel('[rad]')

    axis[2][0].hist(all_state_extra[:,5], bins=50, density=True, color='b', alpha=0.75)
    axis[2][0].set_title('relative_height')
    axis[2][0].set_xlabel('[m]')

    axis[0][1].hist(all_state_extra[:,2], bins=50, density=True, color='b', alpha=0.75)
    axis[0][1].set_title('linear_x')
    axis[0][1].set_xlabel('[m/s]')

    axis[1][1].hist(all_state_extra[:,3], bins=50, density=True, color='b', alpha=0.75)
    axis[1][1].set_title('linear_y')
    axis[1][1].set_xlabel('[m/s]')
    
    axis[2][1].hist(all_state_extra[:,4], bins=50, density=True, color='b', alpha=0.75)
    axis[2][1].set_title('angular_z')
    axis[2][1].set_xlabel('[rad/s]')
    
    fig.tight_layout()
    plt.show()
