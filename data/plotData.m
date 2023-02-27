%% validate data
close all

allTasks = {'pickbox', 'placebox', 'opendraw', 'pulldraw'};
clutter = {'','clutter'};

for j = 1:length(allTasks)
    for c = 1:length(clutter)
        task = [allTasks{j} clutter{c}];
        if contains(task,'draw')
            numDemos = 3;
        else
            numDemos = 6;
        end

        for i = 1:numDemos
            tmp = load(['data/' task num2str(i)],"datapoint");
            img =  tmp.datapoint.img;
            figure;
            imshow(img)
            drawnow
            title([task ' ' num2str(i)])
            pause(1)
        end
    end
end
