"""
Explain affordance network
"""
import os, glob
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from controller import AffordanceController

import torch
from torch import nn
from torchvision import transforms
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget, 
    RawScoresOutputTarget,
)

methods = {
    "gradcam": GradCAM,
    "scorecam": ScoreCAM,
    "gradcam++": GradCAMPlusPlus,
    "ablationcam": AblationCAM,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
    "fullgrad": FullGrad,
}

transform_composed = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
]) 

def calc_cam(image_color_list, cam_algorithm, model, target_layers, targets):
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.

    image_tensor_list = []
    for image_color in image_color_list:
        image_np = image_color.copy()
        image_np = cv2.resize(image_np, (256, 256))
        image_tensor = transform_composed(image_np)
        image_tensor_list.append(image_tensor)

    input_tensor = torch.cat(tuple(image_tensor for image_tensor in image_tensor_list), dim=0)
    input_tensor =  input_tensor.unsqueeze(0)

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=True) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return grayscale_cam

if __name__ == '__main__':

    ### Load Network
    model_param = {
        'afford_model_path': '/media/lab/NEPTUNE2/field_outputs/row_4_10_13/affordance (copy)/affordance_model.pt'
    }
    
    agent_controller = AffordanceController(**model_param)

    ### 
    model = agent_controller.afford_model
    target_layers = [model.net.last_conv_downsample]    
    # target_layers = [model.net.encoder[-1][-1], model.net.last_conv_downsample]
    cam_method = methods['ablationcam']
    # cam_method = methods['gradcam']

    ### Datafolder
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_4/2022-10-14-10-01-08"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_10/2022-10-14-10-41-06"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_12/2022-10-28-13-33-03"
    # folder_path = "/media/lab/NEPTUNE2/field_datasets/row_13/2022-11-06-15-44-34"
    folder_path = "/media/lab/NEPTUNE2/field_datasets/row_18/2022-11-15-11-26-08"

    # image files
    color_file_list = glob.glob(os.path.join(folder_path, 'color', '*.png'))
    color_file_list.sort()

    # Start the loop
    out = cv2.VideoWriter('/home/lab/affordance_distance_abla.avi', cv2.VideoWriter_fourcc(*'MJPG'), 6, (256, 256))     
    for index in range(len(color_file_list)):
        raw_img_list = []
        for j in [0,2,4,10]:
            color_file = color_file_list[max(0, index-j)]
            img_bgr = cv2.imread(color_file, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            raw_img_list.append(img_rgb)

        affordance_pred = agent_controller.predict_affordance(raw_img_list)

        img_rbg_combine = np.concatenate(tuple(cv2.resize(img, (256, 256)) for img in raw_img_list), axis=1)
        img_bgr_combine = cv2.cvtColor(img_rbg_combine, cv2.COLOR_RGB2BGR)

        # cv2.imshow('disp', img_bgr_combine)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q')):
            break

        # # plot
        grayscale_cam = calc_cam(raw_img_list, cam_method, model, target_layers, [ClassifierOutputTarget(0)])
        
        cam_image_list = []
        for rgb_img in raw_img_list:
            rgb_img = cv2.resize(rgb_img, (256, 256))
            rgb_img = np.float32(rgb_img) / 255
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image_list.append(cam_image)
            break

        img_cam_combine = np.concatenate(tuple(img for img in cam_image_list), axis=1)
        img_cam_combine = cv2.cvtColor(img_cam_combine, cv2.COLOR_RGB2BGR)
        out.write(img_cam_combine)
        cv2.imshow('cam', img_cam_combine)

        # time.sleep(1./6.0)
    out.release()