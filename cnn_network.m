%% define network
close all

fdsTrain = fileDatastore('data/', "ReadFcn", @customLoad, "IncludeSubfolders", true);

tmp = read(fdsTrain);
dataPointTemplate = tmp{1};
inputSize = size(dataPointTemplate.x);

% squeezeNet = squeezenet('Weights','imagenet');
% detector = yolov3ObjectDetector("tiny-yolov3-coco");
% net = resnet50('Weights','imagenet');

preLayers = [
    imageInputLayer(inputSize, 'Name','data', 'Normalization','none');
    passThroughLayer
    leakyReluLayer('Name', 'preLayersOut')
%     resize2dLayer('OutputSize', [224 224], 'Name','preLayersOut')
    ];

postLayers = [
%     resize2dLayer('OutputSize', [10 10], 'Method', 'bilinear', 'Name','postLayersIn')
    leakyReluLayer('Name','postLayersIn')
    fullyConnectedLayer(2,'WeightL2Factor',1)];


% lgraph = squeezeNet.layerGraph;
% lgraph = detector.Network.layerGraph;
% lgraph = net.layerGraph;

% names = arrayfun(@(l) l.Name, lgraph.Layers(18:end), 'UniformOutput', false);
% lgraph = lgraph.removeLayers(names);
% lgraph = lgraph.removeLayers('input_1');

layers = [

convolution2dLayer([8,8], 8,'Padding','same', 'Stride', 1,'Name', 'LayersIn')
% maxPooling2dLayer([3 3])
tanhLayer()

resize2dLayer('Scale', .5); 
convolution2dLayer([3,3], 8,'Padding','same', 'Stride', 1)
% maxPooling2dLayer([3 3])
tanhLayer()

% convolution2dLayer([3,3], 8,'Padding','same', 'Stride', 1)
% % maxPooling2dLayer([3 3])
% tanhLayer()

convolution2dLayer([3,3], 8,'Padding','same', 'Stride', 2)
leakyReluLayer('Name', 'LayersOut')

%     convolution2dLayer([50, 50], 1,'Padding','same')
%     leakyReluLayer()

% fullyConnectedLayer(2,'WeightL2Factor',1)

];

lgraph = layerGraph(layers);




lgraph = lgraph.addLayers(preLayers);
lgraph = lgraph.addLayers(postLayers);
lgraph = lgraph.connectLayers('preLayersOut', 'LayersIn');
lgraph = lgraph.connectLayers('LayersOut', 'postLayersIn');



dlnet = dlnetwork(lgraph);

global calcJac
calcJac = false;
global jac

%% training options
close all
calcJac = false;
numEpochs = 1000;
miniBatchSize = 32;
initialLearnRate = 1e-4;
decay = 0.0001;
momentum = 0.9;


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
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        [dlX, dlY] = next(mbq);

        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [loss, gradients, state] = dlfeval(@modelGradients, dlnet,dlX,dlY);
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

%% test
close all

calcJac = true;
jac = [];
dimInd = 1;

fdsTrain.reset()
fdsTrain.shuffle()
for i =1:randi(200)
    tmp = read(fdsTrain);
end
dataPoint = tmp{1};
img = dataPoint.x;
img = dlarray(img ,'SSC');

val = dlfeval(@modelGradientsJac, dlnet, img, dimInd);


figure(2)
subplot(2,2,1)
imshow(extractdata(img))

jacScaled = gather(jac);
for i = 1:3
    subplot(2,2,1+i)
    surf(jacScaled(:,:,i))
end


figure(1)
alpha = -100;
newImg = img;
for i = 1:3
    %     jac = floor(jac*1000)/1000;
    newImg = newImg + alpha*jac;
    valNew = dlfeval(@modelGradientsJac, dlnet, newImg, dimInd);
    valNew-val
end
subplot(1,3,1)
imshow(extractdata(img))
subplot(1,3,2)
imshow(extractdata(newImg))


diffImg = double(gather(extractdata(newImg) - extractdata(img)));
clims = [min(diffImg,[],'all') max(diffImg,[],'all')];
for i = 2:2
    subplot(1,3,2+1)
    imagesc(diffImg(:,:,i))
end




