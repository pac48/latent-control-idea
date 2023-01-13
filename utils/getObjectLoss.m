function loss = getObjectLoss(realPoints2D, nerfPoints2D, meanPenalty, rotPenalty)
realPoints2D = dlarray(gather(extractdata(realPoints2D)), 'SSB');
nerfPoints2DMean = mean(nerfPoints2D,2);
realPoints2DMean = mean(realPoints2D,2);
nerfPoints2DZero = nerfPoints2D - nerfPoints2DMean;
realPoints2DZero = realPoints2D - realPoints2DMean;

tmp = nerfPoints2DMean - realPoints2DMean;
lmin = min([tmp.^2 abs(tmp)], [],2);
loss = meanPenalty*sum(lmin, 'all');

tmp = mean(abs(nerfPoints2DZero - realPoints2DZero),2);
lmin = min([tmp.^2 abs(tmp)], [],2);
loss = loss + rotPenalty*sum(lmin, 'all');

% loss = loss + rotPenalty*sum( (nerfPoints2DZero - realPoints2DZero).^2, 'all');
%     loss = loss./size(realPoints2D,2);
%     loss = loss + loss;
end