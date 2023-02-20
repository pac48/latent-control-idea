function [map, state] = getfullTFNetOutput(fullTFNet, input)
global curInd;
curInd = 1;

in = cell(length(fullTFNet.InputNames), 1);
for i = 1:length(in)
    key = fullTFNet.InputNames{i};
    if isKey(input, key)
        in{i} = input(key);
    else
        layers = fullTFNet.Layers;
        for l = 1:size(layers, 1)
            if strcmp(layers(l).Name, key)
                inputSize = layers(l).InputSize;
                if isa(layers(l), 'nnet.cnn.layer.ImageInputLayer')
                    in{i} = dlarray(ones(inputSize), 'SSCB');
                elseif isa(layers(l), 'nnet.cnn.layer.FeatureInputLayer')
                    in{i} = dlarray(ones(inputSize, 1), 'CB');
                end
            end
        end

    end
end

[map, state] = getNetOutput(fullTFNet, in{:});
matchStr = '_T_world_2_cam';
for key = map.keys
    if contains(key, matchStr)
        val = map(key{1});
        valMod = getT(val(1:3,:), val(4:6,:));
        map(key{1}) = valMod;
    end
end

end