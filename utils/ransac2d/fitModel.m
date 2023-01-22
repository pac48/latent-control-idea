function model = fitModel(combinedPoints)
% combinedPoints: xNerf yNerf xReal yReal

% points = combinedPoints(:,1:2);
% pointsOriginal = combinedPoints(:,3:4);

% combinedPoints = reshape(combinedPoints, size(combinedPoints,1), 2, []);

featuresNerf = combinedPoints(:, 1:end/2);
featuresReal = combinedPoints(:, end/2+1:end);


pointsNerf = featuresNerf(:, 1:2);
pointsReal = featuresReal(:, 1:2);


% points = points';
% pointsOriginal = pointsOriginal';
% cols = cat(2, points', ones(size(points,2),1));
% A = blkdiag(cols, cols);
% b = cat(1, pointsOriginal(1,:)', pointsOriginal(2,:)');
% model = A\b;
%
% model = model';

% tform = fitgeotform2d(points, pointsOriginal, "similarity");
% tform = fitgeotform2d(points, pointsOriginal, "affine");
try
    tform = fitgeotform2d(pointsNerf, pointsReal, "similarity");
    model = tform.A;
catch
    model = [1 0 0
        0 1 0];
end

D = eig(model(1:2, 1:2));
if any(D <= 0)
    model = [1 0 0
        0 1 0];
end

end