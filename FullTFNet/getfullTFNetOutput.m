function [map, state] = getfullTFNetOutput(fullTFNet, img, Z)
global curInd;
curInd = 1;

[map, state] = getNetOutput(fullTFNet, img, Z);
matchStr = '_T_world_2_cam';
for key = map.keys
    if contains(key, matchStr)
        val = map(key{1});
        valMod = getT(val(1:3,:), val(4:6,:));
        map(key{1}) = valMod; 
    end
end

end