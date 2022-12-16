%% test classs
clear all
instr = Instr();
% zed = ZED_Camera(enablePositionTracking=true);
zed = ZED_Camera();
%% get object images
delete('data/masks/*')
delete('data/images/*')
numFiles = length(dir('data'))-2;
close all
while 1
    [left, right] = zed.read_stereo();
    segments = instr.predict(left, right);
    left_scale = double(imresize(left,[480, 640]))./255;
 
    objMasks = getObjMasks(segments);
    if ~isempty(objMasks)
        for i = 1:size(objMasks,3)
            mask = objMasks(:,:,i);
            img = left_scale;
            img = img .*mask;
            numFiles = numFiles + 1;
            imwrite(img, ['data/masks/' num2str(numFiles) '.png'])
            imwrite(left_scale, ['data/images/' num2str(numFiles) '.png'])
%             dataPoint = struct('img', img, 'T', T);
%             save(['data/' num2str(numFiles)], 'T')
        end
    else
        objMasks = zeros(size(segments));
    end


    left_scale = left_scale.*((sum(objMasks,3) ~= 0)+.2);
    imshow(left_scale);
    drawnow
end

%% get background images
delete('data/masks/*')
delete('data/images/*')
numFiles = length(dir('data'))-2;
close all
while 1
    [left, right] = zed.read_stereo();
    left_scale = double(left)./255;
     numFiles = numFiles + 1;
    imwrite(left_scale, ['data/masks/' num2str(numFiles) '.png'])
    imwrite(left_scale, ['data/images/' num2str(numFiles) '.png'])

    imshow(left_scale);
    drawnow

    pause(.2)
end


