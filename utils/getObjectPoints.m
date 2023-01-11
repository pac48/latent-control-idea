function [mkptsNerf, mkptsReal] = getObjectPoints(map, name)
mkptsNerf = []; 
mkptsReal = [];

nerfPointKey = [name '_nerf_ProjectionLayer'];
realPointKey = [name '_nerf_NerfLayer/mkptsReal'];
if ~isKey(map, realPointKey) || ~isKey(map, nerfPointKey) 
    return
end
mkptsNerf = map(nerfPointKey);
mkptsReal = map(realPointKey);

end

