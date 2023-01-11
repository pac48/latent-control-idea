function model = fitModel(combinedPoints)
points = combinedPoints(:,1:2);
pointsOriginal = combinedPoints(:,3:4);

% points = points';
% pointsOriginal = pointsOriginal';
% cols = cat(2, points', ones(size(points,2),1));
% A = blkdiag(cols, cols);
% b = cat(1, pointsOriginal(1,:)', pointsOriginal(2,:)');
% model = A\b;
% 
% model = model';

tform = fitgeotform2d(points, pointsOriginal, "similarity");
model = tform.A;

end