close all
task = {'pick', 'iphone_box'};


demoNum = 4;
tmp = load(['../data/' task{1} task{2} num2str(demoNum) '.mat']);
initMsg = tmp.datapoint.msg(1);

tmp = load("trajs_train_pred_False.mat");

traj = tmp.trajs(demoNum,:,:);
traj = reshape(traj, [], 9); 

tNDP = linspace(0, 5, size(traj,1));
dt = t(2)-t(1);
XNDP = traj;
XdNDP = (XNDP(2:end, :) - XNDP(1:end-1, :) )./dt;
XNDP = XNDP(1:end-1, :);
tNDP = tNDP(1:end-1);


msgs = convertMatrixToMsgsNDP(robot, bodyNames, initMsg, tNDP, XNDP', XdNDP');
playBackDemonstration(robot, msgs)

plot3(XNDP(:, [1 4 7]), XNDP(:, [2 5 8]), XNDP(:, [3 6 9]))


%%
close all
tmp = load(['../data/' task{1} task{2} num2str(demoNum) '.mat']);

figure;plot(XNDP)
X = tmp.datapoint.X;
figure;plot(X')

%%
close all
XNDP = X';
XdNDP = (XNDP(2:end, :) - XNDP(1:end-1, :) )./dt;
XNDP = XNDP(1:end-1, :);
tNDP = tNDP(1:end-1);


msgs = convertMatrixToMsgsNDP(robot, bodyNames, initMsg, tNDP, XNDP', XdNDP');
playBackDemonstration(robot, msgs)

plot3(XNDP(:, [1 4 7]), XNDP(:, [2 5 8]), XNDP(:, [3 6 9]))
