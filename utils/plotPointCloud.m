function plotPointCloud(robot, dlnet, objects, maps, varargin)
assert(length(varargin)<=1);
if isempty(varargin)
    Trobot_cam2 = eye(4);
else
    Trobot_cam2 = varargin{1};
end


% [fl, fx, fy] = getFValues(dlnet);
%  fx = fx*1.2778*.7*1.0405; 
%  fy = fy*1.2778*.7*1.0405;
%  fx = fx*.7; 
%  fy = fy*.7;
% figure
% hold off
% robot.plotObject()

for pInd = 1:length(maps)
    map = maps{pInd};
    %     subplot(1, length(maps), pInd)
    hold on
    
    for i = 1:length(objects)
        object = objects{i};

%         [points, imgtmp] = getObjectPointCloud(map, object, fl, fx, fy);
%         if isempty(points)
%             continue
%         end
     

        [Trobot_object, points, imgtmp, pointsCam] = getObjectTransform(dlnet, map, object, Trobot_cam2);
        if isempty(points)
            continue
        end

%         if  ~strcmp(object, 'background')
%             Trobot_object = eye(4);
%             Trobot_object(1:3, end) = mean(points,2);
%             Tcam2_object = extractdata(map([object '_nerf_T_world_2_cam']));
%             Trobot_object(1:3, 1:3) = Trobot_cam1(1:3, 1:3)*Tcam2_object(1:3, 1:3);
%         else
%             Trobot_object = eye(4);
%         end


        plotTF(Trobot_object, '-')


        cData = permute(imgtmp, [3 1 2]);
        cData = reshape(cData, 3, [])';
        scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)

% %         figure
% %         Tcam2_object = extractdata(map([object '_nerf_T_world_2_cam']));
% %         Tobject_cam2 = inv(Tcam2_object); 
% %         points2 = Tobject_cam2(1:3,1:3)*pointsCam + Tobject_cam2(1:3,end);
% % 
% %         plotTF(eye(4), '-')
% %         plotTF(Tobject_cam2, '--')
% %         cData = permute(imgtmp, [3 1 2]);
% %         cData = reshape(cData, 3, [])';
% %         scatter3(points2(1, :), points2(2, :), points2(3, :), 'CData', cData)

        
    end
end

axis equal
% camproj('perspective')

end