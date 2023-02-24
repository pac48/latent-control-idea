function loss = getAllignmentLoss(map, objects, varargin)
loss = 0;

plotStuff = false;
if ~isempty(varargin)
    plotStuff = varargin{1};
    rowInd = varargin{2};
    numSubRows = varargin{3};
end
if plotStuff
    figure(99)
end
for i = 1:length(objects)
    object = objects{i};
    keycam2world = [object '_nerf_T_world_2_cam'];
    keyBaseT = [object '_nerf_NerfLayer/baseT' ];

    Tcam2world = map(keycam2world);
    baseT = extractdata(map(keyBaseT));
    if isempty(Tcam2world)
        continue
    end

    camPoint = map([object '_nerf_TFLayer/points_cam']);
    if mean(camPoint(3, :)) > -.1
        loss = loss + 1*sum((Tcam2world - inv(baseT)).^2,'all');
        continue;
    end
    

    [mkptsNerf, mkptsReal] = getObjectPoints(map, object);
    if numel(mkptsReal) > 2 && strcmp(object, 'background')
        loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 250/3, 10000/3);
    elseif numel(mkptsReal) > 2
        loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 25, 2000);
    end

    if plotStuff && ~isempty(mkptsNerf)
        subplot(numSubRows,length(objects), (rowInd-1)*length(objects) + i)
        mkptsNerf = extractdata(mkptsNerf);
        mkptsReal = extractdata(mkptsReal);
        hold off
        plot(mkptsNerf(1,:), mkptsNerf(2,:), 'LineStyle','none', 'Marker','.')
        hold on
        plot(mkptsReal(1,:), mkptsReal(2,:), 'LineStyle','none', 'Marker','.')
    end


end
if plotStuff
figure(90)
end

end