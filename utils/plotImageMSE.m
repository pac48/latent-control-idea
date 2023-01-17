function plotImageMSE(fullTFNet, imgs)
numImages = size(imgs, 4);

for ind = 1:numImages
    figure
    subplot(1,3,1)
    image(extractdata(imgs(:,:,:,ind)))

    subplot(1,3,2)
    imgRender = plotRender(fullTFNet, ind);
    imgRender = double(imgRender)./255;

    subplot(1,3,3)
    tmpImg = extractdata(imgs(:,:,:,ind));
    %     base = .0001+sqrt(sum(tmpImg.^2, 3)).*sqrt(sum(imgRender.^2, 3));
    %     corr = sum(tmpImg.*imgRender, 3)./base;
    corr = 1-sum(abs(tmpImg - imgRender),3)./3;
    %     corr = corr.^8;
    corr = corr -.5;
    corr(corr < 0) = 0;
    corr = corr*(1/max(corr,[],"all"));

    imshow(corr)
    drawnow

end

end