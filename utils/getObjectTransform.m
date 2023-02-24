function [Trobot_object, points, imgtmp] = getObjectTransform(dlnet, map, object, Trobot_cam2)
% points: nerf points in camera coordinates
Trobot_object = []; 

Tcam2_cam1 = eye(4);
Tcam2_cam1(1:3, 1:3) = [0 0 -1
                        -1 0 0
                        0 1 0];

Trobot_cam1 = Trobot_cam2*Tcam2_cam1;

[fl, fx, fy] = getFValues(dlnet);

keyDepth = [object '_nerf_NerfLayer/depth'];
keyImg = [object '_nerf_NerfLayer/imgNerf'];
if ~map.isKey(keyImg) || ~map.isKey(keyDepth)
    return
end
depth = map(keyDepth);
img = map(keyImg);

[points, imgtmp] = getObjectPointCloud(img, depth, object, fl, fx, fy);
if isempty(points)
    return
end

points = Trobot_cam1(1:3, 1:3)*points + Trobot_cam1(1:3, end);

if  ~strcmp(object, 'background')
%     Trobot_object = eye(4);
%     Trobot_object(1:3, end) = mean(points, 2);
    Tcam2_object = extractdata(map([object '_nerf_T_world_2_cam']));
%     Trobot_object(1:3, 1:3) = Trobot_cam1(1:3, 1:3)*Tcam2_object(1:3, 1:3);
    Trobot_object = Trobot_cam1*Tcam2_object;
else
    Trobot_object = eye(4);
end


end