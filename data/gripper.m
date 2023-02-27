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


%% set force
msgGrpper.Op = "set";
msgGrpper.Args = '{"signals": {"holding_force_n": {"data": [0.1], "format": {"type": "float"}}}}';
msgGrpper.Time = rostime('now');
pub.send(msgGrpper)


