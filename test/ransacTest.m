%% least sqaures
close all
addpath('ransac2d/')
% clc
% % load pointsForLineFitting.mat
% plot(points(:,1),points(:,2),'o');
% hold on

pointsOriginal = .1*rand(10, 2);
angle = 2*pi*rand(1);
offset = [.4; .4]+.2*rand(2,1);%rand(2,1);
R = eul2rotm([angle 0 0]);
points = (R(1:2, 1:2)*pointsOriginal' + offset)';
% add noise
numCorrupted = 3;
points(1:numCorrupted, :) = points(1:numCorrupted, :) + .1*rand(size(points(1:numCorrupted, :)));
% R(1:2, 1:2)
% offset

% model = fitModel([points(numCorrupted+1:end,:), pointsOriginal(numCorrupted+1:end,:)]);
model = fitModel([points, pointsOriginal]);
subplot(1,2,1)
plotStuff(model, pointsOriginal, points, numCorrupted)

%% RANSAC
clc

sampleSize = 3; % number of points to sample per trial
maxDistance = .0001; % max allowable distance for inliers

fitLineFcn = @fitModel; % fit function using polyfit
evalLineFcn = @fitValue;
modelRANSAC = ransac2d(points, pointsOriginal, sampleSize, maxDistance);

subplot(1,2,2)
plotStuff(modelRANSAC, pointsOriginal, points, numCorrupted)


[R_leastSquare, offset_leastSquare] = getTransform(model);
[R_leastSquare offset_leastSquare]

[R_RANSAC, offset_RANSAC] = getTransform(modelRANSAC);
[R_RANSAC offset_RANSAC]

[R(1:2, 1:2) offset]

% y = evaluateModel(modelRANSAC, points);
% scatter(y(:, 1), y(:, 2), 'green')


%
% modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
%
% inlierPts = points(inlierIdx,:);
% x = [min(inlierPts(:,1)) max(inlierPts(:,1))];
% y = modelInliers(1)*x + modelInliers(2);
% plot(x, y, 'g-')
% legend('Noisy points','Least squares fit','Robust fit');
% hold off

% function model = fitModel(combinedPoints)
% points = combinedPoints(:,1:2);
% pointsOriginal = combinedPoints(:,3:4);
% 
% points = points';
% pointsOriginal = pointsOriginal';
% cols = cat(2, points', ones(size(points,2),1));
% A = blkdiag(cols, cols);
% b = cat(1, pointsOriginal(1,:)', pointsOriginal(2,:)');
% model = A\b;
% 
% model = model';
% 
% end
% 
% function cost = fitValue(model, combinedPoints)
% points = combinedPoints(:,1:2);
% pointsOriginal = combinedPoints(:,3:4);
% cost = sum((pointsOriginal - evaluateModel(model, points) ).^2, 2);
% % pointsOriginal - evaluateModel(model, points)
% 
% end
% 
% function modelPoints = evaluateModel(model, points)
% points = points';
% model = model';
% 
% cols = cat(2, points', ones(size(points,2),1));
% A = blkdiag(cols, cols);
% b = A*model;
% 
% modelPoints = reshape(b, [], 2)';
% modelPoints = modelPoints';
% 
% end
% 
% function plotStuff(model, pointsOriginal, points, numCorrupted)
% scatter(pointsOriginal(:, 1), pointsOriginal(:, 2))
% hold on
% scatter(points(:, 1), points(:, 2),'blue')
% scatter(points(1:numCorrupted, 1), points(1:numCorrupted, 2), 'red')
% 
% y = evaluateModel(model, points);
% scatter(y(:, 1), y(:, 2), 'green', 'Marker','.')
% scatter(y(1:numCorrupted, 1), y(1:numCorrupted, 2), 'red', 'Marker','.')
% 
% end
% 
% 
% function [R, offset] = getTransform(model)
% R = reshape(model([1 2 4 5]),2,2);
% offset = -R*model([3 6])';
% end
% 
% function [modelRANSAC, inlierIdx] = ransac2d(points, pointsOriginal, sampleSize, maxDistance)
% fitLineFcn = @fitModel; % fit function using polyfit
% evalLineFcn = @fitValue;
% 
% [modelRANSAC, inlierIdx] = ransac([points, pointsOriginal], fitLineFcn,evalLineFcn, ...
%     sampleSize,maxDistance, 'MaxNumTrials',1000, 'MaxSamplingAttempts', 1000);
% end