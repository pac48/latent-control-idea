close all

layers = fullTFNet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        break;
    end
end

for v = [0 45]
    hold on
    angle = -90 + v;
    % T = [ -0.6697    0.7184    0.1883    0.0696
    %     -0.5459   -0.6481    0.5310   -0.0468
    %     0.5035    0.2528    0.8262   -0.6183
    %     0         0         0    1.0000];

    T = [0.7997    0.6004    0.0028    0.3572
        -0.1956    0.2561    0.9467   -0.1431
        0.5677   -0.7576    0.3222   -0.8463
        0         0         0    1.0000];

    T(1:3,end) = T(1:3,end);% - T(1:3, 3);


    T = inv(T); % cam to world
    % T(1:3, 1:3) = T(1:3, 1:3)*axang2rotm([0 0 1 pi*angle/180]);
    T(1,end) = T(1,end) + .2*(v-45/2)/45;

    angle2 = 2*(rand(1)-.5)*30;
    R = axang2rotm([1 1 1 pi*angle2/180]);
    T(1:3, 1:3) = R*T(1:3, 1:3);
    T(1:3, end) = R*T(1:3, end);

    renderTest(layer, T, v==0);
end

function renderTest(layer, T, color)
global nerf
global fov

nerf.setTransform(layer.objNames{1}, T)
[imgNerf, depth] = nerf.renderObject(layer.height, layer.width, fov, layer.objNames{1});
imgNerf = uint8(255*imgNerf);



[X,Y] = meshgrid(linspace(-.5, .5, layer.width), linspace(-.5, .5, layer.height));
xPix = X;
yPix = -Y;
[fl, fx, fy] = layer.getFValues();
% inds = [reshape(X,[],1) reshape(Y,[],1)];
% inds = (inds(:,1,: )-1)*layer.height + inds(:,2,:);
% d = depth(inds);
d = reshape(depth,[],1);


xPix = reshape(xPix,[],1);
yPix = reshape(yPix,[],1);
Z = -d;
X = -(xPix).*Z*fx*.8;
Y = -(yPix).*Z*fy*.8;


points = [X Y Z];

points = points';  % camera coordinates
pointsCam = points;
points = pagemtimes(T(1:3, 1:3, :), points) + T(1:3, end, :); % world coordinates



figure(1)
if color
    cData = permute(imgNerf, [3 1 2]);
    cData = reshape(cData, 3, [])';
    scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)
else
    scatter3(points(1, :), points(2, :), points(3, :))
end
axis equal


figure
cData = permute(imgNerf, [3 1 2]);
cData = reshape(cData, 3, [])';
scatter3(pointsCam(1, :), pointsCam(2, :), pointsCam(3, :), 'CData', cData)
axis equal


end