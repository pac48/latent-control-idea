%% test classs
clear all
loftr = LoFTR();
%%
zed = ZED_Camera();

close all
[img0, ~] = zed.read_stereo();
% img0 = imresize(img0, [480 640]);
while 1
    [img1, ~] = zed.read_stereo();
    %     img1 = imresize(img1, [480 640]);
    % 480 640

    [mkpts0, mkpts1, mconf] = loftr.predict(img0, img1);
    if isempty(mkpts0)
        continue
    end

    img = cat(2, img0, img1);
    hold off
    imshow(img);
    hold on
    px = [mkpts1(:,1), mkpts0(:,1)+size(img0, 2)];
    py = [mkpts1(:,2), mkpts0(:,2)];
    plot(px', py',LineWidth=1);
    drawnow

    %     left_scale = double(imresize(left,[480, 640]))./255;
    %     left_scale = left_scale.*(segments ~= 0);
    %     pause(.1)
    %     imshow(left_scale);
end
%%
% img0 = imread('real_img.png');
img0 = imread('new_nerf_1.png');
% img0 = imread('3.png');
% img0 = imresize(img0, [480 640]*.5);
img0 = imresize(img0, [480 640]);
% img1 = imread('nerf_img.png');
img1 = imread('new_nerf_2.png');
% img1 = imread('3_mask.png');
% img1 = imresize(img1, [480 640]*.5);
img1 = imresize(img1, [480 640]);



%%
close all
[mkpts0, mkpts1, mconf] = loftr.predict(img0, img1);
plotCorrespondence(img0, img1, mkpts0, mkpts1, mconf)
% for i = 1:size(px,1)
%     plot3(px(i), py(i), pz(i));
% end

