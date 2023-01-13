function filteredInds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf)
w = size(imgNerf, 2);
h = size(imgNerf, 1);
s = RandStream('mlfg6331_64');
filteredInds = [];


imReal = imgaussfilt(imReal, 3); 
imgNerfOrig = imgNerf;
imgNerf = imgaussfilt(imgNerf, 3);

score = 0;
counter = 0;
bestScore = score;
while score < .85
    if counter == 10
%         bestScore
        return
    end
    counter = counter + 1;
    inds = datasample(s, 1:size(mkptsReal,1), 3, 'Replace', false);
    model = fitModel([mkptsNerf(inds ,:) mkptsReal(inds,:)] - [w/2 h/2 w/2 h/2]); % model predicts real poins from nerf points
    [R, offset] = getTransform(model);
    
%     if norm(R) > 5 || norm(R) < 1/50
%         continue
%     end
    
    A = eye(3);
    A(1:2, 1:2) = R;
    A(1:2, end) = offset;
    
    tform = affinetform2d(A);
    centerOutput = affineOutputView(size(imgNerf), tform, "BoundsStyle","CenterOutput");
    imgNerfRot = imwarp(imgNerf, tform, 'OutputView', centerOutput);
    imgNerfOrigRot = imwarp(imgNerfOrig, tform, 'OutputView', centerOutput);
    inds = any(imgNerfOrigRot > 0, 3);
    
    imgNerfRot = ((double(imgNerfRot)./255)-.5).*inds;
    imRealMod = ((1/255)*double(imReal)-.5).*inds;
    
    base = .001 + sqrt(sum(imRealMod.^2, 3)).*sqrt(sum(imgNerfRot.^2, 3));
    R = sum(imRealMod.*imgNerfRot, 3)./base;

    R = 1-sum(abs(imRealMod - imgNerfRot),3)./3;
    tmp = R(inds);
    percentiles = prctile(tmp, 10);
    vals = tmp(tmp >= percentiles(1));
    score = mean(vals);
    bestScore = max(bestScore, score);
%         figure(2)
%         subplot(1,3,1)
%         imshow(imRealMod+.5)
%         subplot(1,3,2)
%         imshow(imgNerfRot+.5)
%         subplot(1,3,3)
%         imshow(R.*inds)
   
end
if false
figure(2)
subplot(1,3,1)
imshow(imRealMod+.5)
subplot(1,3,2)
imshow(imgNerfRot+.5)
subplot(1,3,3)
imshow(R)
end

R = A(1:2, 1:2);
fitVals = (R*(mkptsNerf - [w/2 h/2])' + offset)' - mkptsReal + [w/2 h/2];
fitVals = sqrt(sum(fitVals(:,1).^2 + fitVals(:,2).^2, 2));
filteredInds = fitVals < maxDistance;

end