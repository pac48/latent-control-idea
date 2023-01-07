"""
Demo script.
"""

import os
import argparse
import time

import cv2
import torch
import numpy as np
from predictor import Predictor

import torch.utils.model_zoo as model_zoo
import torch.onnx

# import pyzed.sl as sl
from ZMQ_server import ZMQServer


# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

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

    if args.save:
        print(f"Saving images to {args.save_dir}")
        os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'left'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'right'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'overlay'), exist_ok=True)

    # load net
    net = Predictor(state_dict_path=args.state_dict, focal_length=args.focal_length, baseline=args.baseline,
                    return_depth=True if args.aux_modality == 'depth' else False)

    # init zed
    cam = Camera()

    ctr = 0
    # main forward loop
    while 1:
        left, right = cam.get_stereo()

        with torch.no_grad():
            pred_segmap, pred_depth = net.predict(left, right)

        if args.viz:
            left = cv2.resize(left, (640, 480), interpolation=cv2.INTER_LINEAR)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('right', cv2.resize(right.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
            cv2.imshow('pred', left_overlay)
            cv2.imshow(args.aux_modality, pred_depth / pred_depth.max())
            cv2.waitKey(1)

        if args.save:
            cv2.imwrite(os.path.join(args.save_dir, 'left', str(ctr).zfill(6) + '.png'), left)
            cv2.imwrite(os.path.join(args.save_dir, 'right', str(ctr).zfill(6) + '.png'), right)
            np.save(os.path.join(args.save_dir, 'depth', str(ctr).zfill(6) + '.npy'), pred_depth)
            cv2.imwrite(os.path.join(args.save_dir, 'segmap', str(ctr).zfill(6) + '.png'), pred_segmap)
            left_overlay = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0), rgb=left, alpha=args.alpha)
            cv2.imwrite(os.path.join(args.save_dir, 'overlay', str(ctr).zfill(6) + '.png'), left_overlay)

            ctr += 1


net = None


def load_model(checkpoint):
    # checkpoint = './pretrained_instr/models/pretrained_model.pth'
    focal_length = 1390.0277099609375 / (2208 / 640)
    baseline = 0.12
    print(checkpoint)
    net = Predictor(state_dict_path=checkpoint, focal_length=focal_length, baseline=baseline, return_depth=True)
    return net


def predict(left_image, right_image):
    global net
    if not net:  # Load model
        net = load_model()

    with torch.no_grad():
        pred_segmap, pred_depth = net.predict(left_image, right_image)

    return torch.from_numpy(pred_segmap).unsqueeze(0)


if __name__ == '__main__':
    # load net

    server = ZMQServer(5555, 'instr')
    state_dict = './pretrained_instr/models/pretrained_model.pth'
    focal_length = 1390.0277099609375 / (2208 / 640)
    baseline = 0.12
    net = Predictor(state_dict_path=state_dict, focal_length=focal_length, baseline=baseline, return_depth=True)

    while True:
        arr = server.recv()
        left = arr[:, :int(arr.shape[1] / 2), :]
        right = arr[:, int(arr.shape[1] / 2):, :]
        # cv2.imshow('left', cv2.resize(left.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
        # cv2.imshow('right', cv2.resize(right.copy(), (640, 480), interpolation=cv2.INTER_LINEAR))
        # cv2.waitKey(1)

        with torch.no_grad():
            pred_segmap, pred_depth = net.predict(left, right)
            print("predicted!")

        # arr = net.colorize_preds(torch.from_numpy(pred_segmap).unsqueeze(0))
        server.send(pred_segmap)


    # load_model('/home/paul/Downloads/instr/pretrained_instr/models/pretrained_model.pth')
    # demo()
    # load_model('/home/paul/Downloads/instr/pretrained_instr/models/pretrained_model.pth')
    # load_model('/home/paul/Downloads/instr/pretrained_instr/models/pretrained_model22.pth')
