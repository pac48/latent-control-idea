function playBackDemonstration(robot, allMsg)
gripperBaseInd = 18;
i = 1;
tStart = double(allMsg(1).Header.Stamp.Sec) + double(allMsg(1).Header.Stamp.Nsec)*1E-9;
tic
while i < length(allMsg)
    ti = toc;
    msg = allMsg(i);
    while double(msg.Header.Stamp.Sec) + double(msg.Header.Stamp.Nsec)*1E-9 - tStart < ti
        msg = allMsg(i);
        i = i + 1;
        if i > length(allMsg)
            return
        end
    end
    

    robot.setJointsMsg(msg);
    hold off
    robot.plotObject
    hold on

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

    drawnow
%     pause(.1)

end

end