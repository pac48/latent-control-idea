function [allZ, Tall] = TFProcessData(objects, dataFolder)
if strcmp(dataFolder, 'pretrain')
    [fileNames, file_path] = TFGetPretrainDataPath();
else
    [fileNames, file_path] = TFGetDataPath();
end

fds = fileDatastore(file_path{1}, "ReadFcn", @load);
data = fds.readall();

indMap = containers.Map(objects, num2cell(1:length(objects)));
TCellData = cell(length(data), length(objects));
% ZCellData = cell(length(data), length(objects));
allZ = [];

for i = 1:length(data)
    dataPoint = data{i}.dataPoint;
    ind = indMap(dataPoint.name);
    TCellData{i, ind} = dataPoint.T;
    allZ(:, i) = dataPoint.Z;

end

allZ = dlarray(allZ, 'CB');

values = cell(1, length(objects));
for i = 1:length(objects)
    values{i} = TCellData(:, i)';
end
Tall = containers.Map(objects, values);


end