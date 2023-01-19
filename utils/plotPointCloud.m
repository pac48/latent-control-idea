function plotPointCloud(robot, dlnet, objects, maps, varargin)
assert(length(varargin)<=1);
if isempty(varargin)
    T = eye(4);
else
    T = varargin{1};
end


[fl, fx, fy] = getFValues(dlnet);

figure
robot.plotObject()

for pInd = 1:length(maps)
    map = maps{pInd};
    %     subplot(1, length(maps), pInd)
    hold on
    
    for i = 1:length(objects)
        object = objects{i};

        [points, imgtmp] = getObjectPointCloud(map, object, fl, fx, fy);
        
        points = T(1:3, 1:3)*points + T(1:3, end);

        cData = permute(imgtmp, [3 1 2]);
        cData = reshape(cData, 3, [])';
        scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)
        axis equal
    end
end


end