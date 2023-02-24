function loss = getPSMLoss(map, target)
loss = 0;

if isempty(target.keys)
    return
end

allKeys = target.keys;
for ss = 1:length(allKeys)
skillName = allKeys{ss};

psmKey = ['PSMLayer_' skillName '/psm'];
psmDDKey = ['PSMLayer_' skillName '/psmDD'];
axisKey = ['PSMLayer_' skillName '/axis'];
Trobot_goalKey = ['PSMLayer_' skillName '/Trobot_goal'];

if ~map.isKey(psmKey) || ~map.isKey(Trobot_goalKey) || ~target.isKey(skillName)
    return
end

psm = map(psmKey);
psmDD = map(psmDDKey);
axis = map(axisKey);

Trobot_goal = map(Trobot_goalKey);
Tgoal_robot = invTransform(Trobot_goal);

psmTargetData = target(skillName);
t = psmTargetData{1};
X = psmTargetData{2};
Xd = psmTargetData{3};

Xprime = transformPoints(X, Tgoal_robot);
Xdprime = transformPointsVel(Xd, Tgoal_robot);

Xprime = permute(Xprime, [2 1 3]);
Xdprime = permute(Xdprime, [2 1 3]);

% PSM interp1
% why
assert(size(Xprime,3)==1);

scale = 3;

totalDisplacement = sqrt(sum((X(1:3, end, :) - X(1:3, 1, :)).^2, 1));
displacementLoss = mean( ( scale*(Xprime(end, 1, :) - Xprime(1, 1, :)) - scale*totalDisplacement ).^2, 'all');
originLoss = sum((scale*Xprime(end, 1:3, :)-0).^2, 'all');
loss = loss + originLoss + displacementLoss;

% scale = 10;

% psmInterp = psm;

cond = diff(extractdata(Xprime(:,1,:))) > 0;
while ~all(cond)
    goodInds = find(cond, 999999999999);
    Xprime = Xprime(goodInds, :, :);
    Xdprime = Xdprime(goodInds, :, :);
    cond = diff(extractdata(Xprime(:,1,:))) > 0;
end

XprimeInterp = [];
XdprimeInterp = [];

if size(Xprime, 1) < 3
    return
end

for i = 1:size(Xprime, 2)
    tmp = interp1(stripdims(Xprime(:,1,:)), stripdims(Xprime(:,i,:)), stripdims(axis)', 'nearest', 'extrap');
    XprimeInterp = cat(2, XprimeInterp, tmp);
    tmp = interp1(stripdims(Xprime(:,1,:)), stripdims(Xdprime(:,i,:)), stripdims(axis)', 'nearest', 'extrap');
    XdprimeInterp = cat(2, XdprimeInterp, tmp);
end




lossVel = mean( (scale*XdprimeInterp(:,1,:) - scale*psm(:,1,:)).^2, 'all');

lossPos = mean( (scale*XprimeInterp(:,2:end,:) - scale*psm(:,2:end,:)).^2, 'all');
lossEnd = 10*mean( (scale*XprimeInterp(end,2:end,:) - scale*psm(end,2:end,:)).^2, 'all');

lossAcc =  .01*mean((psmDD(:,2:end,:)).^2, 'all');

loss = loss + lossVel + lossAcc + lossPos + lossEnd;% + lossPosD;
% loss = sum((Tgoal_robot-1).^2, 'all');

% xd1hat = psm(1, :);

if false
    close all
    figure
    plot(axis, XprimeInterp(:,2:end,:), '--')
    hold on
    plot(axis, psm(:,2:end,:), '-')

    figure
    plot(Xprime(:,1,:), Xdprime(:,1,:), '--')
    hold on
    plot(axis, psm(:,1,:), '-')
end

end

lossPoint = loss


end

function Tgoal_robot = invTransform(Trobot_goal)
R = Trobot_goal(1:3, 1:3, :);
R = stripdims(permute(R, [2 1 3]));
offset = Trobot_goal(1:3, end, :);
p = pagemtimes(-R, stripdims(offset));
ONE = ones(1, 1, size(R,3));
ZERO = zeros(1, 1, size(R,3));

Tgoal_robot = [[R p];[ZERO ZERO ZERO ONE]];
end

function Xprime = transformPoints(X, Tgoal_robot)
Xcell = cell(size(X, 1)/3, 1);
for i = 1:length(Xcell)
    Xcell{i} = X((1:3) + (i-1)*3, :);
end
Xprime = cat(3, Xcell{:}); % this is in robot coordinates
Xprime = pagemtimes(Tgoal_robot(1:3, 1:3, :), Xprime) + Tgoal_robot(1:3, end, :);

for i = 1:length(Xcell)
    Xcell{i} = Xprime(:, :, i);
end
Xprime = cat(1, Xcell{:});
end
function Xprime = transformPointsVel(X, Tgoal_robot)
Xcell = cell(size(X, 1)/3, 1);
for i = 1:length(Xcell)
    Xcell{i} = X((1:3) + (i-1)*3, :);
end
Xprime = cat(3, Xcell{:}); % this is in robot coordinates
Xprime = pagemtimes(Tgoal_robot(1:3, 1:3, :), Xprime);

for i = 1:length(Xcell)
    Xcell{i} = Xprime(:, :, i);
end
Xprime = cat(1, Xcell{:});
end