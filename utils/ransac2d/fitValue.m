function cost = fitValue(model, combinedPoints)
points = combinedPoints(:,1:2);
pointsOriginal = combinedPoints(:,3:4);
cost = sum(abs(pointsOriginal - evaluateModel(model, points) ), 2);
% pointsOriginal - evaluateModel(model, points)

end
