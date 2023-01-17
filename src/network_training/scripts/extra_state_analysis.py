"""
Analyze Extra State Effect 
"""
import os
import glob
import numpy as np
import cv2
import pandas
import matplotlib.pyplot as plt

from controller.vae_latent_control import VAELatentController_Full, VAELatentController, VAELatentController_TRT

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


if __name__ == '__main__':
    # Datafolder
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter0/2022-12-15-09-35-34'
    folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-02-56'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-24'
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/iter1/2022-12-24-13-03-00'

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
        'vae_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/vanilla_vae/vanilla_vae_model_z_1000.pt',
        'latent_model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter1/latent_ctrl/latent_ctrl_vanilla_vae_model_z_1000.pt',
        'model_weight_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter1/combined_vae_latent_ctrl_z_1000.pt',
        'tensorrt_engine_path': '/home/lab/catkin_ws/src/neptune-ros/model_weight/vae/combined_vae_latent_ctrl_z_1000.trt',
    }

    controller_agent = VAELatentController(**model_config)
    controller_agent_full = VAELatentController_Full(**model_config)

    i = int(0.5 * len(data_dict['color_file_list']))
    print("index = ", i)

    image_bgr = cv2.imread(data_dict['color_file_list'][i], cv2.IMREAD_UNCHANGED)

    # state extra
    state_extra = np.array([
        roll_angle[i],
        pitch_angle[i],
        body_linear_x[i],
        body_linear_y[i],
        body_angular_z[i],
        relative_height[i],
    ])

    print("state_extra = ", state_extra)

    # predict
    predicted_action = controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra)
    image_raw, image_pred = controller_agent_full.reconstruct_image(image_bgr, is_bgr=True)

    if abs(predicted_action) < 1e-2:
        predicted_action = 0.0



    # extra travesal
    total_num = 1000
    roll_travesal = np.random.normal(state_extra[0], 0.05, total_num)
    pitch_travesal = np.random.normal(state_extra[1], 0.01, total_num)
    linear_x_travesal = np.random.normal(state_extra[2], 0.02, total_num)
    linear_y_travesal = np.random.normal(state_extra[3], 0.05, total_num)
    angular_z_travesal = np.random.normal(state_extra[4], 0.15, total_num)
    rel_height_travesal = np.random.normal(state_extra[5], 0.08, total_num)
    
    results_roll = []
    results_pitch = []
    results_linear_x = []
    results_linear_y = []
    results_angular_z = []
    results_rel_height = []
    for j in range(total_num):
        state_extra = np.array([
            roll_travesal[j],
            pitch_angle[i],
            body_linear_x[i],
            body_linear_y[i],
            body_angular_z[i],
            relative_height[i],
        ])
        results_roll.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

        state_extra = np.array([
            roll_angle[i],
            pitch_travesal[j],
            body_linear_x[i],
            body_linear_y[i],
            body_angular_z[i],
            relative_height[i],
        ])
        results_pitch.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

        state_extra = np.array([
            roll_angle[i],
            pitch_angle[i],
            linear_x_travesal[j],
            body_linear_y[i],
            body_angular_z[i],
            relative_height[i],
        ])
        results_linear_x.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

        state_extra = np.array([
            roll_angle[i],
            pitch_angle[i],
            body_linear_x[i],
            linear_y_travesal[j],
            body_angular_z[i],
            relative_height[i],
        ])
        results_linear_y.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

        state_extra = np.array([
            roll_angle[i],
            pitch_angle[i],
            body_linear_x[i],
            body_linear_y[i],
            angular_z_travesal[j],
            relative_height[i],
        ])
        results_angular_z.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

        state_extra = np.array([
            roll_angle[i],
            pitch_angle[i],
            body_linear_x[i],
            body_linear_y[i],
            body_angular_z[i],
            rel_height_travesal[j],
        ])
        results_rel_height.append(controller_agent_full.predict(image_bgr, is_bgr=True, state_extra=state_extra))

    results_roll = np.array(results_roll)
    results_pitch = np.array(results_pitch)
    results_linear_x = np.array(results_linear_x)
    results_linear_y = np.array(results_linear_y)
    results_angular_z = np.array(results_angular_z)
    results_rel_height = np.array(results_rel_height)


    fig, axis = plt.subplots(3,2)

    # axis[0][0].hist(roll_travesal, bins=50, density=True, color='b', alpha=0.75)
    # axis[1][0].hist(pitch_travesal, bins=50, density=True, color='b', alpha=0.75)
    # axis[2][0].hist(rel_height_travesal, bins=50, density=True, color='b', alpha=0.75)

    # axis[0][1].hist(linear_x_travesal, bins=50, density=True, color='b', alpha=0.75)
    # axis[1][1].hist(linear_y_travesal, bins=50, density=True, color='b', alpha=0.75)
    # axis[2][1].hist(angular_z_travesal, bins=50, density=True, color='b', alpha=0.75)

    axis[0][0].scatter(roll_travesal, results_roll, 5)
    axis[0][0].scatter(roll_angle[i], predicted_action, 10, c='r')
    axis[0][0].set_title('roll_angle')
    axis[0][0].set_xlabel('[rad]')
    axis[0][0].set_ylim([-1, 1])

    axis[1][0].scatter(pitch_travesal, results_pitch, 5)
    axis[1][0].scatter(pitch_angle[i], predicted_action, 10, c='r')
    axis[1][0].set_title('pitch_angle')
    axis[1][0].set_xlabel('[rad]')
    axis[1][0].set_ylim([-1, 1])

    axis[2][0].scatter(rel_height_travesal, results_rel_height, 5)
    axis[2][0].scatter(relative_height[i], predicted_action, 10, c='r')
    axis[2][0].set_title('rel_height')
    axis[2][0].set_xlabel('[m]')
    axis[2][0].set_ylim([-1, 1])

    axis[0][1].scatter(linear_x_travesal, results_linear_x, 5)
    axis[0][1].scatter(body_linear_x[i], predicted_action, 10, c='r')
    axis[0][1].set_title('linear_x')
    axis[0][1].set_xlabel('[m/s]')
    axis[0][1].set_ylim([-1, 1])

    axis[1][1].scatter(linear_y_travesal, results_linear_y, 5)
    axis[1][1].scatter(body_linear_y[i], predicted_action, 10, c='r')
    axis[1][1].set_title('linear_y')
    axis[1][1].set_xlabel('[m/s]')
    axis[1][1].set_ylim([-1, 1])

    axis[2][1].scatter(angular_z_travesal, results_angular_z, 5)
    axis[2][1].scatter(body_angular_z[i], predicted_action, 10, c='r')
    axis[2][1].set_title('angular_z')
    axis[2][1].set_xlabel('[rad/s]')
    axis[2][1].set_ylim([-1, 1])
    
    fig.tight_layout()
    plt.show()

    # # plot
    # image_raw = cv2.resize(image_raw, (320, 240))
    # image_pred = cv2.resize(image_pred, (320, 240))
    # image_combine = np.concatenate((image_raw, image_pred), axis=1)
    # cv2.imshow('VAE', cv2.cvtColor(image_combine, cv2.COLOR_RGB2BGR))

    # cv2.waitKey(0)