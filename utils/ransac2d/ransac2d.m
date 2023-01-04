function [modelRANSAC, inlierIdx] = ransac2d(points, pointsOriginal, sampleSize, maxDistance)
fitLineFcn = @fitModel; % fit function using polyfit
evalLineFcn = @fitValue;

[modelRANSAC, inlierIdx] = ransac([points, pointsOriginal], fitLineFcn,evalLineFcn, ...
    sampleSize,maxDistance, 'MaxNumTrials',1000, 'MaxSamplingAttempts', 1000);
end









