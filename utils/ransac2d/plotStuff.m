function plotStuff(model, pointsOriginal, points, numCorrupted)
scatter(pointsOriginal(:, 1), pointsOriginal(:, 2))
hold on
scatter(points(:, 1), points(:, 2),'blue')
scatter(points(1:numCorrupted, 1), points(1:numCorrupted, 2), 'red')

y = evaluateModel(model, points);
scatter(y(:, 1), y(:, 2), 'green', 'Marker','.')
scatter(y(1:numCorrupted, 1), y(1:numCorrupted, 2), 'red', 'Marker','.')

end