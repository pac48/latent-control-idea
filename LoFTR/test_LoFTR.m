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
img0 = imread('boxRealTestCropped.jpg');
% img0 = imread('real_img.png');
% img0 = imread('new_nerf_1.png');
% img0 = imread('3.png');
% img0 = imresize(img0, [480 640]*.5);
img0 = imresize(img0, [480 640]*4);
% img1 = imread('nerf_img.png');
% img1 = imread('new_nerf_2.png');
% img1 = imread('3_mask.png');
% img1 = imread('real_img_gimp.png');
img1 = imread('boxNerfTestCropped2.jpg');
% img1 = imresize(img1, [480 640]*.5);
img1 = imresize(img1, [480 640]*4);



% img0 = cat(4, img0, img0, img0);
% img1 = cat(4, img1, img1, img1);

%%
close all
[mkpts0, mkpts1, mconf] = loftr.predict(img0, img1);
plotCorrespondence(img0, img1, mkpts0, mkpts1, mconf)
% for i = 1:size(px,1)
%     plot3(px(i), py(i), pz(i));
% end

%%  Find the SURF features.
% load('imgNerf.mat')
% load('imReal.mat')
load('imgNerfBest.mat')
load('imRealBest.mat')

img0 = imRealBest;
% img0 = imresize(img0, 2);
img1 = imgNerfBest;
% img1 = imresize(img1, 2);

I1 = rgb2gray(img0);
I2 = rgb2gray(img1)
% I2(I2==0) = I1(I2==0);

points1 = detectHarrisFeatures(I1)
points2 = detectHarrisFeatures(I2);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2, 'MatchThreshold', 100) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));


figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
legend("matched points 1","matched points 2");

