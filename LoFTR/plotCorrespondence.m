function plotCorrespondence(img0, img1, mkpts0, mkpts1)
    [X,Y] = meshgrid(0:(size(img1,2)-1), 0:(size(img1,1)-1));
    Z = ones(size(X));
    hold off
    surf(X, Y, Z, 'CData', img1, 'edgecolor', 'none');
    hold on
    [X,Y] = meshgrid(0:(size(img0,2)-1), 0:(size(img0,1)-1));
    Z = ones(size(X));
    surf(X, Y, -Z, 'CData', img0, 'edgecolor', 'none')
    
    px = [mkpts0(:,1) mkpts1(:,1)];
    py = [mkpts0(:,2) mkpts1(:,2)];
    pz = [mkpts1(:,2)*0-1 mkpts1(:,2)*0+1.1];
    plot3(px', py', pz',LineWidth=2);
end