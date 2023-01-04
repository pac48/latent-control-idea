function modelPoints = evaluateModel(model, points)
points = points';
model = model';

cols = cat(2, points', ones(size(points,2),1));
A = blkdiag(cols, cols);
b = A*model;

modelPoints = reshape(b, [], 2)';
modelPoints = modelPoints';

end