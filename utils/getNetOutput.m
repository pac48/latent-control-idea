function [map, state] = getNetOutput(dlnet, Z, img)
values = cell(1, length(dlnet.OutputNames));
[values{:}, state] = dlnet.predict(Z, img);
keys = dlnet.OutputNames;
map = containers.Map(keys, values);

end

