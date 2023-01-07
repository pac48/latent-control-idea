close all
clear all

addpath('LoFTR/')
addpath('SuperGlue/')
addpath('nerf/')
addpath('utils/')
addpath('utils/ransac2d/')

loftr = LoFTR();
nerf = Nerf({'nerf_background', 'nerf_box', 'nerf_cup'});

fov = 101.2768;
height = 480;
width = 640;
% numBackgroundTransforms = length(nerf.name2Frame('nerf_background'));
% numBoxTransforms = length(nerf.name2Frame('nerf_box'));
% numCupTransforms = length(nerf.name2Frame('nerf_cup'));
%% correspondence
close all

[allT, allImgs] = nerf.name2Frame('nerf_cup');
T1 = allT{10};
nerf.setTransform({'nerf_cup', T1});
[img1, depth1] = nerf.renderObject(height, width, fov, 'nerf_cup');
T2 = T1;
tic
% while 1
% T2(1:3, end) = T2(1:3, end) + T2(1:3, 2)*2*.01*toc;
T2(1:3, end) = T2(1:3, end) - T2(1:3, 1)*2;
nerf.setTransform({'nerf_cup', T2});
[img2, depth2] = nerf.renderObject(height, width, fov, 'nerf_cup');
% imshow(img2)
% drawnow()
% end
figure(1)
subplot(1,2,1)
imshow(img1)
subplot(1,2,2)
imshow(img2)

[mkptsReal, mkptsNerf, mconf] = loftr.predict(uint8(255*img1), uint8(255*img2));

maxDistance = 3;
sampleSize = 3;
[modelRANSAC, inlierIdx] = ransac2d(mkptsNerf, mkptsReal, sampleSize, maxDistance);
mkptsNerf = mkptsNerf(inlierIdx, :);
mkptsReal = mkptsReal(inlierIdx, :);
mconf = mconf(inlierIdx);

figure(2)
plotCorrespondence(img1, img2, mkptsReal, mkptsNerf, mconf);

%% world points
inds = floor(mkptsNerf);
inds = (inds(:,1,: )-1)*height + inds(:,2,:);
goodInds = depth2(inds) ~= 0;
inds = inds(goodInds);
if isempty(inds)
    error('not suppose to happen')
end
%             inds = (1:numel(depth2))';
%             tmp = depth2(inds) ~= 0;
%             inds = inds(tmp);
d = depth2(inds);

[X,Y] = meshgrid(linspace(-.5, .5, width), linspace(-.5, .5, height));
xPix = X(inds);
yPix = -Y(inds);
fl = 1;
fx = fl*2*tand(fov/2);
fy = fx*height/width;
xDir =  fx*xPix;
yDir =  -fy*yPix;
zDir = fl*ones(size(yDir));
vec = [xDir yDir zDir];
vec = vec./sqrt(sum(vec.^2, 2));
points = vec.*d;
points = points';  % camera coordinates
points = pagemtimes(T2(1:3, 1:3, :), points) + T2(1:3, end, :); % world coordinates


%                         figure
%                         imgR = img2(:,:,1);
%                         imgG = img2(:,:,2);
%                         imgB = img2(:,:,3);
%                         imgR = imgR(inds);
%                         imgG = imgG(inds);
%                         imgB = imgB(inds);
%                         img = cat(3, imgR, imgG, imgB);
%                         cData = permute(img, [3 1 2]);
%                         cData = reshape(cData, 3, [])';
%                         scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)
%                         axis equal


mkptsNerf = [mkptsNerf(goodInds,1) mkptsNerf(goodInds,2)];
mkptsReal = [mkptsReal(goodInds,1) mkptsReal(goodInds,2)];
mconf = mconf(goodInds);

%% LEAST SQAURE SOLUTION TO FIND fx, fy, and cx, cy!!!!!!
inds = (1:numel(depth2))';
tmp = depth2(inds) ~= 0;
inds = inds(tmp);
d = depth2(inds);
[X,Y] = meshgrid(linspace(-.5, .5, width), linspace(-.5, .5, height));
xPix = X(inds);
yPix = Y(inds);
fl = 1;
fx = fl*2*tand(fov/2);
fy = fx*height/width;
xDir =  fx*xPix;
yDir =  fy*yPix;
zDir = fl*ones(size(yDir));
vec = [xDir yDir zDir];
vec = vec./sqrt(sum(vec.^2, 2));
pointsCal = vec.*d;
pointsCal = pointsCal';  % camera coordinates
% pointsCal = pagemtimes(T2(1:3, 1:3, :), pointsCal) + T2(1:3, end, :); % world coordinates

figure
imgR = img2(:,:,1);
imgG = img2(:,:,2);
imgB = img2(:,:,3);
imgR = imgR(inds);
imgG = imgG(inds);
imgB = imgB(inds);
img = cat(3, imgR, imgG, imgB);
cData = permute(img, [3 1 2]);
cData = reshape(cData, 3, [])';
scatter3(pointsCal(1, :), pointsCal(2, :), pointsCal(3, :), 'CData', cData)
axis equal

X = pointsCal(1,:);
Y = pointsCal(2,:);
Z = pointsCal(3,:);
% [x,y]' = [f 0 cx; 0 f cy; 0 0 1] [X,Y,Z, 1]'
Ax = [X'./Z' zeros(length(X),1) ones(length(X),1) zeros(length(X),1)];
bx = width*(xPix+.5);
Ay = [zeros(length(X),1) Y'./Z' zeros(length(X),1) ones(length(X),1)];
by = height*(yPix+.5);

A = cat(1, Ax, Ay);
b = cat(1, bx, by);
w = A\b

fl_x = w(1);
fl_y = w(2);
cx = w(3);
cy = w(4);


%% n-point
worldPoints = points';
% 
% offset = [width height];
% mkptsRealNormalized = [fx fy].*(mkptsReal)./offset;

imagePoints = mkptsReal;

% fl_x = 262.49218021053844;
% fl_y = 350.42178907953246;
% cx = 321.5289056246761;
% cy = 240.87673067768395;

% fl_x = fx;
% fl_y = fy;
% cx = fx/2;
% cy = fy/2;

focalLength = [fl_x, fl_y];%floor(100*[fx, fy]); % specified in units of pixels
principalPoint =  [cx, cy]; % in pixels [x, y]
imageSize = [height, width]; % in pixels [mrows, ncols]

intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);
intrinsics.K

worldPose = estworldpose(imagePoints, worldPoints, intrinsics)


% figure(3)
% pcshow(worldPoints,VerticalAxis="Y",VerticalAxisDir="down", ...
%     MarkerSize=30);
% hold on
% plotCamera(Size=.1,Orientation=worldPose.R', ...
%     Location=worldPose.Translation);
% hold off

% transform = eye(4);
% transform(1:3, :) = [worldPose.R worldPose.Translation']

% % transform(1:3, end) = T1(1:3,end); 
% % transform(1:3, 1:3) = T1(1:3,1:3); 
% % transform = T1;



imagePoints2 = mkptsNerf;
% [params,imgsUsed,Errors] = estimateCameraParameters(imagePoints, imagePoints2)
camExtrinsics = estimateExtrinsics(imagePoints2, imagePoints, intrinsics);
transform = eye(4);
transform(1:3, :) = [camExtrinsics.R camExtrinsics.Translation'./1000]


% pointsCam = pagemtimes(transform(1:3, 1:3, :), points) + transform(1:3, end, :); % world coordinates
% pointsCam2d = [pointsCam(1,:); pointsCam(2,:)]./pointsCam(3,:);
% scatter(pointsCam2d(1,:), pointsCam2d(2,:)); 
% hold on
% scatter(imagePoints(:, 1), imagePoints(:, 2))
% xlim([-.5*fx .5*fx])
% ylim([-.5*fy .5*fy])


%% render new pose
figure(4)

[allT, allImgs] = nerf.name2Frame('nerf_cup');
% T2 = T1;
% T2(1:2, end) = T2(1:2, end) + [1; 1];


T2New = T2*transform
% T2New = transform;

nerf.setTransform({'nerf_cup', T2New});
[img2, depth2] = nerf.renderObject(height, width, fov, 'nerf_cup');

subplot(1,2,1)
imshow(img1)
subplot(1,2,2)
imshow(img2)


