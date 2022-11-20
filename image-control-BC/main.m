% scene = Scene('model/model.dae');
% 
% num_targets = 1000;
% ind = 0;
% goal_pose = scene.meshes(3).transform(1:3,end);
% 
% for i = 1:num_targets
% 
% %     uncomment for more variation
%     goal_pose = scene.meshes(3).transform(1:3,end);
%     goal_pose(1:2) = 5*(rand(2,1)-.5);
%     scene.meshes(3).transform(1:3,end) = goal_pose;
% 
% %     figure(1)
%     im = render(scene);
%     img = gather(im);
%     img = imresize(img,.25);
% %     imshow(img)
% 
%     data_point = struct('img', img, 'pos', goal_pose);
%     save(fullfile('data', num2str(ind)),"data_point")
% 
%     ind = ind+1;
%     continue
% 
%     cur_pose = scene.meshes(3).transform(1:3,end);
%     cur_pose(1:2) = 5*(rand(2,1)-.5);
%     scene.meshes(3).transform(1:3,end) = cur_pose;
% 
% 
% 
% 
% 
% 
%     while norm(goal_pose-cur_pose) > .05
%         ind = ind+1;
%         tic
%         cur_pose = scene.meshes(3).transform(1:3,end);
%         new_pose = cur_pose + .2*(goal_pose-cur_pose);
% 
%         scene.meshes(3).transform(1:3,end) = new_pose ;
% 
% %         figure(2)
%         im = render(scene);
%         img = gather(im);
%         img = imresize(img,.25);
% %         imshow(img)
%         drawnow
%         toc
% 
%         data_point.cur = img;
%         data_point.action = new_pose-cur_pose;
% 
%         save(fullfile('data', num2str(ind)),"data_point")
% 
%     end
% end

%% define network
close all

% fdsTrain = fileDatastore('data/', "ReadFcn", @customLoad, "IncludeSubfolders", true);
fdsTrain = fileDatastore('data/', "ReadFcn", @customLoad, "IncludeSubfolders", true);

tmp = read(fdsTrain);
dataPointTemplate = tmp{1};
inputSize = size(dataPointTemplate.img);
inputSize(3) = inputSize(3);

arrayY = linspace(-1, 1, 50)';
arrayX = linspace(-1, 1, 50)';

[X,Y] = meshgrid(linspace(-1, 1, 64), linspace(-1, 1, 64) );
XFlat = reshape(X, 1, []);
YFlat = reshape(Y, 1, []);

XFlat = repmat(XFlat,1,1);
YFlat = repmat(YFlat,1,1);

layers = [
    imageInputLayer(inputSize, 'Name','data', 'Normalization','none');

        convolution2dLayer([5,5], 8, 'Stride', 2,'Name', 'LayersIn')
    leakyReluLayer

    convolution2dLayer([5,5], 16, 'Stride', 1)
    leakyReluLayer

    convolution2dLayer([5,5], 32, 'Stride', 1)
    leakyReluLayer

    convolution2dLayer([5,5], 64,'Stride', 1)
    leakyReluLayer

%         convolution2dLayer([3,3], 64,'Stride', 1)
%     leakyReluLayer

%     convolution2dLayer([3,3], 32,'Stride', 1)
%     leakyReluLayer

%     convolution2dLayer([3,3], 64, 'Stride', 1)
%     leakyReluLayer

%     convolution2dLayer([3,3], 128, 'Stride', 1)
%     leakyReluLayer

    % convolution2dLayer([3,3], 128, 'Stride', 1)
    % leakyReluLayer
    %
    % convolution2dLayer([3,3], 256, 'Stride', 1)
    % leakyReluLayer



    %
    % convolution2dLayer([3,3], 32,'Padding','same', 'Stride', 1)
    % tanhLayer
    %
    %
    % convolution2dLayer([3,3], 32,'Padding','same', 'Stride', 1)
    % tanhLayer
    %
    convolution2dLayer([3,3], 1, 'Padding','same', 'Stride', 1)
    leakyReluLayer('Name', 'Out')

    resize2dLayer('OutputSize', [64,64], 'Method','bilinear')

    %     fullyConnectedLayer(2*length(arrayX))
    %     ReshapeLayer([2, length(arrayX)], 'Reshape')
    flattenLayer
    functionLayer(@(W) abs(W))%.^2)
% %     softmaxLayer
    %     functionLayer(@(W) [pagemtimes(W(1,:,:), arrayX); pagemtimes(W(2,:,:), arrayY)] )
    functionLayer(@(W) [XFlat*W; YFlat*W] )
    fullyConnectedLayer(2)

    ];

lgraph = layerGraph(layers);
dlnet = dlnetwork(lgraph);

%% training options
close all
calcJac = false;
numEpochs = 1000;
miniBatchSize = 8;
initialLearnRate = 1e-4;
decay = 0.0001;
momentum = 0.98;


figure(1)
h1 = subplot(1,3,1);
h2 = subplot(1,3,2);
h3 = subplot(1,3,3);
lineLossTrain = animatedline(h1, 'Color',[0.85 0.325 0.098]);
ylim(h1, [0 inf])
xlabel(h1, "Iteration")
ylabel(h1, "Loss")
grid(h1,'on')

%% train
mbq = minibatchqueue(fdsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn',@(Xcell) preprocessMiniBatch(Xcell),...
    'MiniBatchFormat',{'SSCB','CB'},...
    'NumOutputs',2);

velocity = [];


iteration = 0;
iterationLast = 0;
start = tic;

loss = 100;
% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        %         if rand < .05%loss < .01 || iteration-iterationLast > 100
        [dlX, dlY] = next(mbq);
        %             iterationLast = iteration;
        %         end

        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [loss, gradients, state] = dlfeval(@modelGradients, dlnet, dlX, dlY);
        %         loss
        dlnet.State = state;

        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);

        % Update the network parameters using the SGDM optimizer.
        [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);

        if mod(iteration, 2)==0
            % Display the training progress.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain, iteration, loss)
            title(h1, "Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end

save('dlnet', "dlnet")
%%
scene = Scene('model/model.dae');



while 1
    goal_pose = scene.meshes(3).transform(1:3,end);
    goal_pose(1:2) = 5*(rand(2,1)-.5);
    scene.meshes(3).transform(1:3,end) = goal_pose;

    figure(1)
    im = render(scene);
    img = gather(im);
    img = imresize(img,.25);
    imshow(img)

    figure(2)
    % [dlX, dlY] = next(mbq);

    X = dlX(:,:,:,1);
    X(:,:,:) = img;


    cur_pose = scene.meshes(3).transform(1:3,end);
    cur_pose(1:2) = 5*(rand(2,1)-.5);
    scene.meshes(3).transform(1:3,end) = cur_pose;

    data_point = struct('goal', img, 'cur', [], 'action', []);
tic


    while norm(goal_pose-cur_pose) > .1 && toc < 4
%         ind = ind+1;
        

        im = render(scene);
        img = gather(im);
        img = imresize(img,.25);
        imshow(img)
        %         X(:,:,1:3) = img;

        [Y_pred, state] = forward(dlnet, X);
        Y_pred
        cur_pose = scene.meshes(3).transform(1:3,end);
        new_pose(1:2) = cur_pose(1:2) + .5*(gather(extractdata(Y_pred))-cur_pose(1:2));
        %         new_pose = cur_pose(1:2) + 1*gather(extractdata(Y_pred));

        scene.meshes(3).transform(1:2,end) = new_pose;

        %         figure(2)

        drawnow
        



    end

end