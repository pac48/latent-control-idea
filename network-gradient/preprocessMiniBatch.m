function [X, Y] = preprocessMiniBatch(XCell)
% Preprocess predictors.
images = cellfun( @(dataPoint) dataPoint.x, XCell, 'UniformOutput', false);

% X = cat(4, images{1:end});
s = size(images{1});
X = zeros(s(1),s(2),s(3), length(XCell));
for i = 1:length(XCell)
X(:,:,:,i) = images{i};
end

points = cellfun( @(dataPoint) dataPoint.y', XCell, 'UniformOutput', false);
Y = cat(2, points{1:end});

end