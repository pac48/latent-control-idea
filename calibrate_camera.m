cam = webcam(1);
%%
% cam.ExposureMode = 'manual';
% cam.Exposure = 100;

% cam.Resolution = '640x480';

% cam.Gain = 16;
cam.Resolution 
imshow(cam.snapshot)