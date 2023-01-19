function [mkptsNerf, mkptsReal] = getObjectPoints(map, name)
mkptsNerf = []; 
mkptsReal = [];

nerfPointKey = [name '_nerf_TFLayer/points_cam'];
realPointKey = [name '_nerf_NerfLayer/mkptsReal'];
if ~isKey(map, realPointKey) || ~isKey(map, nerfPointKey) 
    return
end
pointNerf = map(nerfPointKey);

fl = 1;
mkptsNerf = projectPoints(pointNerf, fl);
mkptsReal = map(realPointKey);

end

