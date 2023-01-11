addpath('LoFTR/')
addpath('SuperGlue/')
addpath('nerf/')
addpath('utils/')
addpath('utils/ransac2d/')

load('blue_block_test_data.mat')
% load('blue_block_test_data2.mat')
% load('book_test_data.mat')

%% least sqaures
close all

figure(1)
plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf, mconf);

maxDistance = 20;
inds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf);

figure(3)
plotCorrespondence(imReal, imgNerf, mkptsReal(inds,:), mkptsNerf(inds,:), mconf(inds));


% function inds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf)
% w = size(imgNerf, 2);
% h = size(imgNerf, 1);inds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf)
% w = size(imgNerf, 2);
% h = size(imgNerf, 1);
% s = RandStream('mlfg6331_64');
% inds = [];
% 
% score = 0;
% counter = 0;
% while score < .9
%     if counter == 100
%         return
%     end
%     counter = counter + 1;
%     filteredInds = datasample(s, 1:size(mkptsReal,1), 3,'Replace',false);
%     model = fitModel([mkptsNerf(filteredInds,:) mkptsReal(filteredInds,:)] - [w/2 h/2 w/2 h/2]); % model predicts real poins from nerf points
%     [R, offset] = getTransform(model);
%     
%     if norm(R) > 5
%         continue
%     end
%     
%     A = eye(3);
%     A(1:2, 1:2) = R;
%     A(1:2, end) = offset;
%     
%     tform = affinetform2d(A);
%     centerOutput = affineOutputView(size(imgNerf), tform, "BoundsStyle","CenterOutput");
%     imgNerfRot = imwarp(imgNerf, tform, 'OutputView', centerOutput);
%     inds = any(imgNerfRot > 0, 3);
%     
%     imgNerfRot = ((double(imgNerfRot)./255)-.5).*inds;
%     imRealMod = ((1/255)*double(imReal)-.5).*inds;
%     
%     base = .01+sqrt(sum(imRealMod.^2, 3)).*sqrt(sum(imgNerfRot.^2, 3));
%     R = sum(imRealMod.*imgNerfRot, 3)./base;
%     tmp = R(inds);
%     percentiles = prctile(tmp, 50);
%     vals = tmp(tmp >= percentiles(1));
%     score = mean(vals);
%     %     figure(2)
%     %     subplot(1,3,1)
%     %     imshow(imRealMod+.5)
%     %     subplot(1,3,2)
%     %     imshow(imgNerfRot+.5)
%     %     subplot(1,3,3)
%     %     imshow(R)
%    
% end
% 
% % figure(2)
% % subplot(1,3,1)
% % imshow(imRealMod+.5)
% % subplot(1,3,2)
% % imshow(imgNerfRot+.5)
% % subplot(1,3,3)
% % imshow(R)
% 
% R = A(1:2, 1:2);
% fitVals = (R*(mkptsNerf - [w/2 h/2])' + offset)' - mkptsReal + [w/2 h/2];
% fitVals = sqrt(sum(fitVals(:,1).^2 + fitVals(:,2).^2, 2));
% inds = fitVals < maxDistance;
% 
% end
% s = RandStream('mlfg6331_64');
% inds = [];
% 
% score = 0;
% counter = 0;
% while score < .9
%     if counter == 100
%         return
%     end
%     counter = counter + 1;
%     filteredInds = datasample(s, 1:size(mkptsReal,1), 3,'Replace',false);
%     model = fitModel([mkptsNerf(filteredInds,:) mkptsReal(filteredInds,:)] - [w/2 h/2 w/2 h/2]); % model predicts real poins from nerf points
%     [R, offset] = getTransform(model);
%     
%     if norm(R) > 5
%         continue
%     end
%     
%     A = eye(3);
%     A(1:2, 1:2) = R;
%     A(1:2, end) = offset;
%     
%     tform = affinetform2d(A);
%     centerOutput = affineOutputView(size(imgNerf), tform, "BoundsStyle","CenterOutput");
%     imgNerfRot = imwarp(imgNerf, tform, 'OutputView', centerOutput);
%     inds = any(imgNerfRot > 0, 3);
%     
%     imgNerfRot = ((double(imgNerfRot)./255)-.5).*inds;
%     imRealMod = ((1/255)*double(imReal)-.5).*inds;
%     
%     base = .01+sqrt(sum(imRealMod.^2, 3)).*sqrt(sum(imgNerfRot.^2, 3));
%     R = sum(imRealMod.*imgNerfRot, 3)./base;
%     tmp = R(inds);
%     percentiles = prctile(tmp, 50);
%     vals = tmp(tmp >= percentiles(1));
%     score = mean(vals);
%     %     figure(2)
%     %     subplot(1,3,1)
%     %     imshow(imRealMod+.5)
%     %     subplot(1,3,2)
%     %     imshow(imgNerfRot+.5)
%     %     subplot(1,3,3)
%     %     imshow(R)
%    
% end
% 
% % figure(2)
% % subplot(1,3,1)
% % imshow(imRealMod+.5)
% % subplot(1,3,2)
% % imshow(imgNerfRot+.5)
% % subplot(1,3,3)
% % imshow(R)
% 
% R = A(1:2, 1:2);
% fitVals = (R*(mkptsNerf - [w/2 h/2])' + offset)' - mkptsReal + [w/2 h/2];
% fitVals = sqrt(sum(fitVals(:,1).^2 + fitVals(:,2).^2, 2));
% inds = fitVals < maxDistance;
% 
% end