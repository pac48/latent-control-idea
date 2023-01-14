function T = getObjectTransforms(dlnet, map, name)
T = [];
key = [name '_nerf_T_world_2_cam'];
if ~isKey(map, key)
    return
end
T = map(key);
% for layer = dlnet.Layers'
%     if isa(layer, 'NerfLayer')
%         lName = [name '_nerf_NerfLayer'];
%         if strcmp(layer.Name, lName)
%             j = layer.h.structure.(lName).jBest;
%             if isempty(j)
%                 return
%             end
%             T = out(:,:,j);
%             return
%         end
%     end
% end
end

