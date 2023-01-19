function out = projectPoints(points, fl)
% point: 3d points in camera cordinates (3 x n x batch)
% n is the number of detected key points

out = dlarray([], 'SSB');

if isempty(points)
    return
end

X = points(1,:,:);
Y = points(2,:,:);
Z = points(3,:,:);
x = -fl*X./Z;
y = -fl*Y./Z;

out = cat(1, x, y);
end