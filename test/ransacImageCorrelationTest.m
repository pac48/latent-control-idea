addpath('LoFTR/')
addpath('SuperGlue/')
addpath('nerf/')
addpath('utils/')
addpath('utils/ransac2d/')

% load('blue_block_test_data.mat')
% load('blue_block_test_data2.mat')
% load('book_test_data.mat')
% load('filterIndsDebug.mat')
load('filterIndsDebug2.mat')

%% least sqaures
close all

figure(1)
plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf);

maxDistance = 50;
inds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf);

figure(3)
plotCorrespondence(imReal, imgNerf, mkptsReal(inds,:), mkptsNerf(inds,:));

%%
close all

figure(1)
plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf);


sampleSize = 3;
maxDistance = 20;
% 
% dataReal = reshape(mkptsReal, [], 1, 2);
% nerfReal = reshape(mkptsNerf, [], 1, 2);
% inds = sub2ind(size(imgNerf), floor(mkptsNerf(:, 2)), floor(mkptsNerf(:, 1)) );
nerfColors = zeros(size(mkptsNerf,1), 3);
for i = 1:size(nerfColors, 1)
    nerfColors(i, :) = imgNerf(floor(mkptsNerf(i, 2)), floor(mkptsNerf(i, 1)),:);
end
realColors = zeros(size(mkptsReal, 1), 3);
for i = 1:size(realColors, 1)
    realColors(i, :) = imReal(floor(mkptsReal(i, 2)), floor(mkptsReal(i, 1)),:);
end

data = cat(2, mkptsNerf, nerfColors, mkptsReal, realColors); % xNerf yNerf xReal yReal 

[model, inds] = ransac(data, @fitModel, @fitValue, sampleSize, maxDistance);


figure(3)
plotCorrespondence(imReal, imgNerf, mkptsReal(inds,:), mkptsNerf(inds,:));
