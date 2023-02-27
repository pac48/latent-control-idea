function loss = getBaseTLoss(map, objects, varargin)
loss = 0;

for i = 1:length(objects)
    object = objects{i};
    keycam2world = [object '_nerf_T_world_2_cam'];
    keyBaseT= [object '_nerf_NerfLayer/baseT' ];

    Tcam2world = map(keycam2world);
    baseT = extractdata(map(keyBaseT));
  
    loss = loss + 1*sum((Tcam2world - inv(baseT)).^2,'all');

end


end