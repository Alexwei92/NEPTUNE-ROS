"""
Test VAE Reconstruction Performance
"""
import os
import glob
import numpy as np
import cv2
import pandas
import time

from controller.vae_control import VAECtrl

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

if __name__ == '__main__':
    # Datafolder
    # folder_path = '/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06'
    folder_path = '/media/lab/NEPTUNE2/field_datasets/human_data/2022-11-30-15-09-23'

    # Read data
    data_dict = read_data(folder_path)

    # Load VAE parameter
    model_config = {
        'vae_model_path': '/media/lab/NEPTUNE2/field_outputs/row_4_10_13/vanilla_vae/vanilla_vae_model_z_128.pt',
        # 'latent_model_path': '/media/lab/Extreme SSD/my_outputs/longmap2/latent_nn/iter1/latent_nn/latent_nn_vanilla_vae_model_z_128.pt'
    }

    controller_agent = VAECtrl(**model_config)

    tic = time.time()
    for i in range(len(data_dict['color_file_list'])):
        image_bgr = cv2.imread(data_dict['color_file_list'][i], cv2.IMREAD_UNCHANGED)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        # predict
        image_raw, image_pred = controller_agent.reconstruct_image(image_bgr)
        image_raw = cv2.resize(image_raw, (640, 480))
        image_pred = cv2.resize(image_pred, (640, 480))
        image_combine = np.concatenate((image_raw, image_pred), axis=1)

        # plot
        plot_with_cmd('human', image_bgr, data_dict['control_cmd'][i], True)
        cv2.imshow('raw', cv2.cvtColor(image_combine, cv2.COLOR_RGB2BGR))

        elapsed_time = time.time() - tic
        time.sleep(max(data_dict['timestamp'][i] - elapsed_time, 0))