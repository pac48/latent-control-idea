function [map, state] = getNetOutput(dlnet, varargin)
values = cell(1, length(dlnet.OutputNames));
[values{:}, state] = dlnet.predict(varargin{:});
keys = dlnet.OutputNames;
map = containers.Map(keys, values);

end

