import os
import argparse
import cv2
import torch
import numpy as np
from predictor import Predictor

import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnxruntime

import pyzed.sl as sl
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import onnx
import torchvision.transforms.functional as ttf
from PIL import Image
from utils.utils import pekdict

class Camera:
    def __init__(self, *args, **kwargs):
        # self.vc = cv2.VideoCapture(2)
        # self.vc.set(cv2.CAP_PROP_EXPOSURE, 100)
        # self.vc.set(cv2.CAP_PROP_GAIN, 8)
        # import subprocess
        # subprocess.check_call("v4l2-ctl -d /dev/video2 -c exposure_absolute=10",shell=True)

        # if not self.vc.isOpened():  # try to get the first frame
        #     exit(1)  # fail to open
        self.zed = sl.Camera()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit()

        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 5)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 10)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)

    def get_stereo(self):
        # Grab an image
        image_left = sl.Mat()
        image_right = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image_left, sl.VIEW.LEFT)  # Get the left image
            self.zed.retrieve_image(image_right, sl.VIEW.RIGHT)  # Get the left image

        # rval, frame = self.vc.read()
        # self.vc.grab()
        # retval, frame = self.vc.retrieve(0)
        # left = frame[:, :int(frame.shape[1]/2),:]
        # right = frame[:, int(frame.shape[1]/2):,:]
        left = image_left.get_data()
        right = image_right.get_data()
        left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (640, 480), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("ZED", left)
        # key = cv2.waitKey(5)
        return left[:, :, :3], right[:, :, :3]


def disp_to_depth(disp, f, b):
    disp = disp.squeeze().cpu().numpy().astype(np.float32)
    depth = np.zeros_like(disp)
    depth[disp != 0] = b * f / disp[disp != 0]
    return depth


def preds_to_map(preds):
    preds = torch.sigmoid(preds)

    preds[preds < 0.5] = 0
    # select based on argmax
    valid_maps = preds.argmax(dim=1, keepdim=False)
    # don't automatically select pixels where all channels have predicted 0
    valid_maps[torch.all(preds == 0, dim=1)] = 0

    return valid_maps.cpu().squeeze().numpy()


def process_im(im, device=torch.device('cuda')):
    im = np.array(im)[:, :, :3]
    im = ttf.to_pil_image(im)
    im = ttf.resize(im, [480, 640], interpolation=Image.LINEAR)
    im = ttf.to_tensor(im)
    im = ttf.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return im.unsqueeze(0).to(device=device)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict(net_session, left, right):
    device = torch.device('cuda')
    t_left, t_right = process_im(im=left, device=device), process_im(im=right, device=device)

    # with torch.no_grad():
    #     preds = net({'color_0': t_left, 'color_1': t_right})

    # x = {'color_0': t_left, 'color_1': t_right}

    in_1 = net_session.get_inputs()[0].name
    in_2 = net_session.get_inputs()[1].name
    ort_inputs = {in_1: t_left, in_2: t_right}
    # preds = net_session.run('output', [{in_1: to_numpy(t_left)}, {in_2: to_numpy(t_right)}])
    tmp = {in_1: to_numpy(t_left), in_2: to_numpy(t_right)}
    preds = net_session.run(None, tmp)

    # rets = pekdict()
    # make sure that the final decoder output is always at pos 0 for loss and tb
    # for i, hsd in enumerate(range(self.hs_dim - 1, -1, -1)):
    #     pred = F.interpolate(inst_pred[:, hsd, :, :, :], (480, 640), mode='bilinear', align_corners=True)
    #     rets.add(f'predictions_{i}', pred, tb=_convert_instanceseg_to_grid)
    #     rets.add(f'predictions_map_{i}', pred.clone().detach(), tb=_convert_instanceseg_to_map)

    return preds_to_map(preds) #, disp_to_depth(preds['disp_pred'].cpu().squeeze().numpy())


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, default='./pretrained_instr/models/pretrained_model.pth')
    parser.add_argument('--focal-length', type=float,
                        default=1390.0277099609375 / (2208 / 640))  # ZED intrinsics per default
    parser.add_argument('--baseline', type=float, default=0.12)  # ZED intrinsics per default
    parser.add_argument('--viz', default=True, action='store_true')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--save-dir', type=str, default='./recorded_images')
    parser.add_argument('--aux-modality', type=str, default='depth', choices=['depth', 'disp'])
    parser.add_argument('--alpha', type=float, default=0.4)
    args = parser.parse_args()

    # net = onnx.load("instr.onnx")
    # onnx.checker.check_model(net)

    net_session = onnxruntime.InferenceSession("instr.onnx")

    # init zed
    cam = Camera()

    ctr = 0
    # main forward loop
    while 1:
        left, right = cam.get_stereo()

        # with torch.no_grad():
        pred_segmap, pred_depth = predict(net_session, left, right)

        left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
        # left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
        cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
        cv2.imshow('right', cv2.resize(right.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
        # cv2.imshow('pred', left_overlay)
        cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
        cv2.waitKey(1)

        ctr += 1


if __name__ == '__main__':
    demo()
