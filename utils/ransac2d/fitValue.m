function cost = fitValue(model, combinedPoints)
% combinedPoints: xNerf yNerf (colorsNerf) xReal yReal (colorsNerf)

featuresNerf = combinedPoints(:, 1:end/2);
featuresReal = combinedPoints(:, end/2+1:end);

pointsNerf = featuresNerf(:, 1:2);
pointsReal = featuresReal(:, 1:2);
colorsNerf = featuresNerf(:, 3:end)' - 128;
colorsReal = featuresReal(:, 3:end)' - 128;



% [R, offset] = getTransform(model);
% modelPoints = (R*pointsNerf'+ offset)';

dist = sqrt(sum((pointsReal- evaluateModel(model, pointsNerf)).^2, 2));


xy   = dot(colorsNerf, colorsReal);
nx   = sqrt(sum(colorsNerf.^2, 1));
ny   = sqrt(sum(colorsReal.^2, 1));
nxny = nx.*ny;
Cs = xy./nxny;
Cs(xy==0) = -inf;

colorCost = floor(100*(1-Cs'));
% colorCost = floor(300*(1-Cs'));

% colorCost = sum(abs(colorsNerf - colorsReal)./3, 2);

cost = dist + colorCost;

end
