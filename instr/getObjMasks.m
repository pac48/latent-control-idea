function objMasks = getObjMasks(segments)
objIds = unique(segments);
objMasks = [];

[X,Y] = meshgrid(0:size(segments, 2)-1, 0:size(segments,1)-1);
for i = 2:length(objIds)
    objSegment = segments==objIds(i);
    xPoints = objSegment.*X;
    xPoints = xPoints(objSegment ~= 0)+1;
    yPoints = objSegment.*Y;
    yPoints = yPoints(objSegment ~= 0)+1;

    P = cat(2,xPoints, yPoints);
    if size(P,1) < 400
        %             segments(objSegment) = 0;
        continue
    end
    [k, ~] = convhull(P);
    polyin = polyshape(P(k,:), KeepCollinearPoints=true);
    A = area(polyin);

    quality = length(xPoints)/A;
    if quality < .95
        continue
        %             segments(objSegment) = 0;
    end

    objMasks = cat(3,objMasks,objSegment);
end

end