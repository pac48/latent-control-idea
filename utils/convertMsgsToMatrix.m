function [t, X, Xd] = convertMsgsToMatrix(robot, allMsg, bodyNames, interpLength)

startTime = getMsgTimeAsDouble(allMsg(1));

X = zeros(length(bodyNames)*3, length(allMsg));
for i = 1:length(allMsg)
    msg = allMsg(i);
    robot.setJointsMsg(msg);
    b = 1;
    bodyName = bodyNames{b};
    T = robot.getBodyTransform(bodyName);
    X((1:3) +(b-1)*3, i) = T(1:3, end);
    for b = 2:length(bodyNames)
        bodyName = bodyNames{b};
        T = robot.getBodyTransform(bodyName);
        X((1:3) +(b-1)*3, i) = T(1:3, end) - X(1:3, i);
    end

end

maxTime = getMsgTimeAsDouble(allMsg(end)) - startTime;

Xcell = cell(size(X, 1), 1);
Xdcell = cell(size(X, 1), 1);
t = linspace(0, maxTime, size(X, 2));
for i  = 1:size(X, 1)
    x = X(i, :);
    x = interp1(t, x, linspace(0, maxTime, interpLength));
    Xcell{i} = x;
end

t = linspace(0, maxTime, interpLength);

X = cat(1, Xcell{:});


numBasis = 20;

width = t(end)/2;
c = linspace(-2, t(end)+2, numBasis-1);

A = ones(length(t), numBasis);
A(:, 1:numBasis-1) = radialBasis(t', c, width);

Ad = zeros(length(t), numBasis);
Ad(:, 1:numBasis-1) = radialBasisD(t', c, width);
C = cat(1, A(end,:), Ad(1,:), Ad(end,:));
d = [0;0];

for i  = 1:size(X, 1)
    b = X(i, :);
    dall = cat(1, b(end), d);
    w = lsqlin(A, b, [], [], C, dall);

    x = A*w;
    xd = Ad*w;

    Xcell{i} = x';
    Xdcell{i} = xd';

end

% hold off
% plot(t, X)
X = cat(1, Xcell{:});
Xd = cat(1, Xdcell{:});
% hold on
% plot(t, X)


end

function t = getMsgTimeAsDouble(msg)

sec =  msg.Header.Stamp.Sec;
Nsec = msg.Header.Stamp.Nsec;

t = double(sec) + double(Nsec)*1E-9;

end