function createPretrainData(featureNet, objects)

[fileNames, file_path] = TFGetPretrainDataPath();

paths = cellfun(@(x) fullfile(file_path, x), fileNames);
numFiles = length(paths);

for i = 1:length(objects)
    numFiles = createPretrainDataObject(featureNet, objects{i}, file_path, numFiles);
end

end

function numFiles = createPretrainDataObject(featureNet, preTrainObject, file_path, numFiles)
global nerf

assert(strcmp(featureNet.Layers(69).Name, 'image_input'))
imageSize = featureNet.Layers(69).InputSize(1:2);

tmp = nerf.name2Images(['nerf_' preTrainObject]);
tmp = cellfun(@(x) double(imresize(x, imageSize))./255, tmp, 'UniformOutput', false);
tmp = cellfun(@(x) dlarray(double(x), 'SSCB'), tmp,  'UniformOutput', false);
imgs = cat(4, tmp{:});

T = nerf.name2Frame(['nerf_' preTrainObject]);
for j = 1:length(T)
    T{j} = inv(T{j});
end

Z = double(extractdata(featureNet.predict(imgs)));

for i = 1:length(T)
    dataPoint = struct('name', preTrainObject, 'Z', Z(:,i), 'T', T{i});
    dataPointName = fullfile(file_path{1}, num2str(numFiles + 1));
    save(dataPointName, 'dataPoint')
    numFiles = numFiles + 1;
end

end