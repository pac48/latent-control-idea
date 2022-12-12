%% test classs
instr = Instr(); 
zed = ZED_Camera();

close all
while 1
    [left, right] = zed.read_stereo();
    segments = instr.predict(left, right);
    left_scale = double(imresize(left,[480, 640]))./255;
    left_scale = left_scale.*(segments ~= 0);
    pause(.1)
    imshow(left_scale);
end

