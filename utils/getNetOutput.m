function [map, state] = getNetOutput(dlnet, img, Z)
values = cell(1, length(dlnet.OutputNames));
[values{:}, state] = dlnet.predict(img, Z);
keys = dlnet.OutputNames;
map = containers.Map(keys, values);

end

