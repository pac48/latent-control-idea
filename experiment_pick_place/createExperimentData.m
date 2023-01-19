function createExperimentData
% createData


basePath = '/home/paul/latent-control-idea/experiment_pick_place/';

imd = imageDatastore([basePath 'img/']);
% fd = datastore('traj/', 'ReadFcn', @load, 'Type', 'file');
% matFiles = fd.Files;
imgFiles = imd.Files;
robot = Sawyer();

close all
for i = 1:length(imgFiles)
    figure
    subplot(1,2,1)
    img = imread(imgFiles{i});
    imshow(img)
    tmp = load([basePath '/traj/t' num2str(i) '_1.mat']);
    allMsg = tmp.allMsg;

    subplot(1,2,2)
    robot.setJointsMsg(allMsg(1));
    start = robot.getBodyTransform(18);
    playback(robot, allMsg(end))
    robot.setJointsMsg(allMsg(end));
    goal = robot.getBodyTransform(18);

    dataPoint = struct('img', img, 'start', start, 'goal', goal);
    fileName = [basePath '/data/' num2str(i)];
    save(fileName, 'dataPoint')
end


end


function playback(robot, allMsg)
% playback

gripperBaseInd = 18;
for i = 1:length(allMsg)
    msg = allMsg(i);
    robot.setJointsMsg(msg);
    hold off
    robot.plotObject
    hold on
    % for ind = 1:28
    T = robot.getBodyTransform(gripperBaseInd);
    point = T(1:3,end);
    line =  [point point+T(1:3,1)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r')
    line =  [point point+T(1:3,2)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g')
    line =  [point point+T(1:3,3)*.15];
    hold on
    plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b')
    % end
    drawnow
    pause(.01)

end

end