function [points, colors] =  getObjectPointCloud(img, depth, object, fl, fx, fy)
points = [];
colors = [];

% Tcam_background = extractdata(map('background_nerf_T_world_2_cam'));
% Tbackground_cam = inv(Tcam_background);

% mapScale = containers.Map( {'background', 'book', 'iphone_box'}, {1.6467*.45, 0.4958*.35, 0.4958*.35});

objScale = 0.1538;
mapScale = containers.Map( {'background', 'book', 'iphone_box', 'plate', 'fork', 'blue_block', 'drawer', 'new_plate', 'jug', 'napkin', 'pepper', 'salt'}, ...
    {.5*1.25, objScale, objScale, objScale, objScale, objScale, objScale*1.1, objScale*1.1, objScale*1.1, objScale*1.1, objScale*1.1, objScale*1.1});
% postScale = 0.6745;%1.6467/2.4414;
% postScale = 1.6467/2.4414;
% postScale = .55;

postScale = 1;

% scale = mapScale(object)*0.6745*0.8759*0.9172*1.7705*1.0844*.5*1.1;
scale = mapScale(object);
% scale = 0.6745;
% % scale = .5;

% depth = scale*extractdata(map(keyDepth));
% img = extractdata(map(keyImg));

if isa(depth, 'dlarray')
    depth = extractdata(depth);
end
if isa(img, 'dlarray')
    img = extractdata(img);
end

depth = scale*depth;

if isempty(img)
    return
end
% depth = imresize(depth, .5);
% img = imresize(img, .5);


width = size(img, 2);
height = size(img, 1);

inds = (1:numel(depth))';
d = depth(inds);
[X,Y] = meshgrid(linspace(-.5, .5, width), linspace(-.5, .5, height));
% [X,Y] = meshgrid(linspace(-1, 1, width), linspace(-1, 1, height));
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
X = -(xPix).*Z*fx;
Y = -(yPix).*Z*fy;
points = [X Y Z];

points = points';  % camera coordinates

% points = Tbackground_cam(1:3, 1:3)*points + Tbackground_cam(1:3, end);
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