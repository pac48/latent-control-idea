function [mkptsNerf, mkptsReal] = getObjectPoints(map, name)
mkptsNerf = [];
mkptsReal = [];

nerfPointKey = [name '_nerf_TFLayer/points_cam'];
realPointKey = [name '_nerf_NerfLayer/mkptsReal'];
if ~isKey(map, realPointKey) || ~isKey(map, nerfPointKey)
    return
end
pointNerf = map(nerfPointKey);

% if strcmp(name, 'iphone_box')
d = sqrt(sum(pointNerf.^2, 1));
if any(d < .8)
    warning('too close to camera')
end
% end
% fl = 1;
mkptsNerf = projectPoints(pointNerf);
mkptsReal = map(realPointKey);

end

