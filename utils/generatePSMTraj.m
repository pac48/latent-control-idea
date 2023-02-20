function [t, X, Xd] = generatePSMTraj(robot, bodyNames, map, skillName)

Trobot_goalKey = ['PSMLayer_' skillName '/Trobot_goal'];
axisKey = ['PSMLayer_' skillName '/axis'];
psmKey = ['PSMLayer_' skillName '/psm'];
psmDKey = ['PSMLayer_' skillName '/psmD'];


Trobot_goal = extractdata(map(Trobot_goalKey));
Tgoal_robot = inv(Trobot_goal);

axis = extractdata(map(axisKey));
psm = extractdata(map(psmKey));
psmD = extractdata(map(psmDKey));

xStart = zeros(length(bodyNames), 3);
for b = 1:length(bodyNames)
    bodyName = bodyNames{b};
    T = robot.getBodyTransform(bodyName);
    xStart((1:3) +(b-1)*3) = T(1:3, end);
end

xPrimeStart = Tgoal_robot(1:3, 1:3)*xStart + Tgoal_robot(1:3, end);

% numSample = 200;
% axisInterp = linspace(xPrimeStart(1,1), 0, numSample);

t = [];
X = [];
Xd = [];
xPrime = [];
xdPrime = [];

xPrime1i = xPrimeStart(1,1);
xPrimei = xPrimeStart(2:end);
% xdPrime1i = 0*xPrime1i;
% xdPrimei = 0*xPrimei;

ind = 1;
dt = 0.01;
ti = 0;

if xPrime1i > 0
    s = -1;
else
    s =  1;
end
xPrime1i = s*xPrime1i;

while xPrime1i < 0
    while xPrime1i > axis(ind)
        ind = ind + 1;
    end

    xdPrime1i = psm(ind, 1);
    xdPrime1i = max(xdPrime1i, .05);
    xdPrimei = xdPrime1i*psmD(ind, 2:end);
    
    xPrime1i = xPrime1i + xdPrime1i*dt;
    xPrimei = xPrimei + 1*(psm(ind, 2:end)- xPrimei) + xdPrimei*dt;
    
    ti = ti+dt;

    t = cat(1, t, ti);
    xPrimeiAll = cat(2, s*xPrime1i, xPrimei);
    xdPrimeiAll = cat(2, s*xdPrime1i, xdPrimei);

    Xi = Trobot_goal(1:3, 1:3)*reshape(xPrimeiAll, 3, []) + Trobot_goal(1:3, end);
    Xi = reshape(Xi, 1,[]);
    
    Xdi = Trobot_goal(1:3, 1:3)*reshape(xdPrimeiAll, 3, []);
    Xdi = reshape(Xdi, 1,[]);


    xPrime = cat(1, xPrime, xPrimeiAll);
    xdPrime = cat(1, xdPrime, xdPrimeiAll);
    X = cat(1, X, Xi);
    Xd = cat(1, Xd, Xdi);

end

numSamples = 200;
inds = floor(linspace(1, size(X, 1), numSamples));
% xPrime = xPrime(inds, :)';
X = X(inds, :)';
Xd = Xd(inds, :)';
t = t(inds)';

% psmInterp = [];
% for i  = 2:size(psm, 2)
%     psmi = psm(:, i);
%     tmp = interp1(axis, psmi, axisInterp, 'nearest', 'extrap');
%     psmInterp = cat(2, psmInterp, tmp);
% end

% integrate



end