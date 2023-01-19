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
    key = [object '_nerf_T_world_2_cam'];
    T = map(key);
    if isempty(T)
        continue
    end

    [mkptsNerf, mkptsReal] = getObjectPoints(map, object);
    if numel(mkptsReal) > 2 && strcmp(object, 'background')
        loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 10, 100);
    elseif numel(mkptsReal) > 2
        loss = loss + getObjectLoss(mkptsReal, mkptsNerf, 10, 100);
    end

    if plotStuff
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