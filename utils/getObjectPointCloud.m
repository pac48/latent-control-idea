function [points, colors] =  getObjectPointCloud(map, object, fl, fx, fy)
points = [];
colors = [];

keyDepth = [object '_nerf_NerfLayer/depth'];
keyImg = [object '_nerf_NerfLayer/imgNerf'];
if ~map.isKey(keyImg) || ~map.isKey(keyDepth)
    return
end

Tcam_background = extractdata(map('background_nerf_T_world_2_cam'));
Tbackground_cam = inv(Tcam_background);

mapScale = containers.Map( {'background', 'book', 'iphone_box'}, {0.9*1.3069*0.8386, 0.3011, 0.3011});
% postScale = 0.6745;%1.6467/2.4414;
postScale = 1.0;%1.6467/2.4414;

scale = mapScale(object)*0.6745*0.8759*0.9172*1.7705*1.0844*.5*1.1;
% scale = 0.6745;

depth = scale*extractdata(map(keyDepth));
img = extractdata(map(keyImg));

% depth = imresize(depth, .5);
% img = imresize(img, .5);


width = size(img, 2);
height = size(img, 1);

inds = (1:numel(depth))';
d = depth(inds);
[X,Y] = meshgrid(linspace(-.5, .5, width), linspace(-.5, .5, height));
xPix = X(inds);
yPix = -Y(inds);
% xDir =  fx*xPix;
% yDir =  fy*yPix;
% zDir = -fl*ones(size(yDir));
% vec = [xDir yDir zDir];
% vec = vec./fx;
% vec = vec./sqrt(sum(vec.^2, 2));
% points = vec.*d;

Z = -d;
X = -(xPix).*Z./(1/fx);
Y = -(yPix).*Z./(1/fy);
points = [X Y Z];

points = points';  % camera coordinates

points = Tbackground_cam(1:3, 1:3)*points + Tbackground_cam(1:3, end);
points = postScale*points;
indsCrop = abs(d) < 3 & abs(d) > 0.01;
% indsCrop = abs(d) < 3000000000 & abs(d) > 0.00000001;
points = points(:, indsCrop);


imgR = img(:,:,1);
imgG = img(:,:,2);
imgB = img(:,:,3);
imgR = imgR(indsCrop);
imgG = imgG(indsCrop);
imgB = imgB(indsCrop);
colors = cat(3, imgR, imgG, imgB);

end