function Tobj_cam_map = getObjectsTMap(dlnet, imgs, objects)
Tobj_cam_map = containers.Map(objects, repmat({{}}, 1, length(objects)));
for ind = 1:size(imgs, 4)
    [map, ~] = getNetOutput(dlnet, imgs(:,:,:, ind), dlarray(ind,'CB'));

    for i = 1:length(objects)
        object = objects{i};
        key = [object '_nerf_T_world_2_cam'];
        T = map(key);
        Tobj_cam_map(object) = cat(2, Tobj_cam_map(object), {T});
    end
end

end