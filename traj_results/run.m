%% init ros
rosshutdown()
rosinit('http://192.168.1.10:11311')
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');
jointCommandPub = rospublisher('/robot/limb/right/joint_command', 'DataFormat','struct');

robot = Sawyer();
bodyNames = {'right_electric_gripper_base','right_gripper_l_finger_tip', 'right_gripper_r_finger_tip'};
%%
pub = rospublisher('/io/end_effector/right_gripper/command');
msgGrpper = rosmessage(pub);
%% close
msgGrpper.Op = "set";
msgGrpper.Args = '{"signals": {"position_m": {"data": [0.0], "format": {"type": "float"}}}}';
msgGrpper.Time = rostime('now');

pub.send(msgGrpper)

%% open
msgGrpper.Op = "set";
msgGrpper.Args = '{"signals": {"position_m": {"data": [0.041667], "format": {"type": "float"}}}}';
msgGrpper.Time = rostime('now');
pub.send(msgGrpper)

%%

close all
% task = {'pick', 'iphone_box'};
% task = {'place', 'book'};
% task = {'pull', 'drawer'};
task = {'open', 'drawer'};


demoNum = 3;
tmp = load(['../data/' task{1} task{2} num2str(demoNum) '.mat']);
initMsg = tmp.datapoint.msg(1);
datapoint = tmp.datapoint;


tmp = load("ndp_trajs_draweropen/trajs_train_pred_False.mat");
% tmp = load("ndp_trajs_drawerpulling/trajs_train_pred_False.mat");
% tmp = load("ndp_trajs_placeiphonebox/trajs_train_pred_False.mat");

traj = tmp.trajs(demoNum,:,:);
traj = reshape(traj, [], 9); 

tNDP = linspace(0, 5, size(traj,1));
dt = tNDP(2)-tNDP(1);
XNDP = traj;
XdNDP = (XNDP(2:end, :) - XNDP(1:end-1, :) )./dt;
XNDP = XNDP(1:end-1, :);
tNDP = tNDP(1:end-1);

robot.setJointsMsg(initMsg);
msgs = convertMatrixToMsgsNDP(robot, bodyNames, initMsg, tNDP, XNDP', XdNDP');
playBackDemonstration(robot, msgs)

plot3(XNDP(:, [1 4 7]), XNDP(:, [2 5 8]), XNDP(:, [3 6 9]))
%% execute 1
gain = 2;
time_scale = 2;
is_reverse = false;
th = .3;

executeReference(gain, datapoint.msg, joint_sub, robot, jointCommandPub, [], time_scale, th, true)

%% execute 2

executeReference(gain, msgs, joint_sub, robot, jointCommandPub, [], time_scale, th, is_reverse)


%%
close all


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
