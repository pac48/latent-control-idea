function T = getT(rpy, p)
R = getR(rpy);
ONE = repmat(1, 1,1, size(R,3));
ZERO = repmat(0, 1,1, size(R,3));
T = [[R p]; ZERO ZERO ZERO ONE];
%                 0   0   0   1];

end