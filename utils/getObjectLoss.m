function loss = getObjectLoss(realPoints2D, nerfPoints2D, meanPenalty, rotPenalty)

% nerfPoints2DMean = mean(nerfPoints2D,2);
% realPoints2DMean = mean(realPoints2D,2);
% nerfPoints2DZero = nerfPoints2D - nerfPoints2DMean;
% realPoints2DZero = realPoints2D - realPoints2DMean;
% pointDiff = extractdata(realPoints2DZero-nerfPoints2DZero);
% pointSum = sum(abs(pointDiff), 1);
% percentiles = prctile(pointSum, 90);
% inds = 1:length(pointSum);
% inds = inds(pointSum < percentiles);
% % inds = inds(pointSum <= .1);
% realPoints2D = realPoints2D(:, inds);
% nerfPoints2D = nerfPoints2D(:, inds); 

% 
% numSamples = min(size(nerfPoints2D, 2), 5);
% s = RandStream('mlfg6331_64', 'Seed', 'shuffle');
% inds = datasample(s, 1:size(nerfPoints2D, 2), numSamples, 'Replace', false);
% realPoints2D = realPoints2D(:, inds);
% nerfPoints2D = nerfPoints2D(:, inds); 


% tmp = .001*(realPoints2D - nerfPoints2D).^2;
% tmp = tanh(tmp);
% 
% loss = 1000.0*sum(mean(tmp, 2));
% return


realPoints2D = dlarray(gather(extractdata(realPoints2D)), 'SSB');
nerfPoints2DMean = mean(nerfPoints2D,2);
realPoints2DMean = mean(realPoints2D,2);

nerfPoints2DZero = nerfPoints2D - extractdata(nerfPoints2DMean);
realPoints2DZero = realPoints2D - extractdata(realPoints2DMean);



% tmp = nerfPoints2DMean - realPoints2DMean;
% % tmp = mean(10*abs(realPoints2D - nerfPoints2D), 2);
% lmin = min([tmp.^2 abs(tmp)], [],2);
% lmin = tmp.^2;

% lmin = mean((nerfPoints2D - realPoints2D).^2, 2);
lmin = (nerfPoints2DMean - realPoints2DMean).^2;
loss = meanPenalty*sum(lmin, 'all');

% tmp = mean( abs(nerfPoints2DZero - realPoints2DZero), 2);
% lmin = min([tmp.^2 abs(tmp)], [],2);
% lmin = tmp.^2;

pointDiff = (nerfPoints2DZero - realPoints2DZero);
lmin1 = mean(pointDiff.^2, 2);
lmin2 = mean( abs(pointDiff), 2);

if sum(lmin1, "all") < sum(lmin2, "all")
    loss = loss + rotPenalty*sum(lmin1, 'all');
else
    loss = loss + rotPenalty*sum(lmin2, 'all');
end


% loss = loss + rotPenalty*sum( (nerfPoints2DZero - realPoints2DZero).^2, 'all');
%     loss = loss./size(realPoints2D,2);
%     loss = loss + loss;
end