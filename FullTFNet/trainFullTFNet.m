function fullTFNet = trainFullTFNet(fullTFNet, inputs, targets, objects, thresh)
batchSize = length(inputs);

initialLearnRate = 2e-1;
decay = 0.05;
momentum = 0.85;
velocity = [];

iteration = 0;
index = 0;

loss = 1;
setSkipNerf(fullTFNet, false);

totalGrad = 1000000;


while totalGrad > thresh || iteration < 100
    tic
    %     if mod(iteration, 401) == 0
    %         setSkipNerf(fullTFNet, false);
    %     else

    %     end

    [loss, gradients, state] = dlfeval(@FullTFModelGradients, fullTFNet, inputs, targets, objects);
    loss
    if isempty(gradients)
        break;
    end
    gradients = thresholdL2Norm(gradients, .3, objects);

    decayFactor = 1/(1 + decay*iteration);
    learnRate = initialLearnRate*decayFactor;
    [fullTFNet, velocity] = sgdmupdate(fullTFNet, gradients, velocity, learnRate, momentum);

    if mod(iteration, 10000000000000000) == 0
        iteration
        index = mod(index + 1, batchSize);
        subplot(1,1,1)
        plotAllCorrespondence(fullTFNet, index+1)
    end

    setSkipNerf(fullTFNet, true);

    iteration = iteration + 1;
    toc

    totalGrad = 0;
    for i = 1:size(gradients,1)
        row = gradients(i, :);
        if strcmp(row.Parameter, "Bias")
            val = row.Value;
            val = extractdata(val{1});
            totalGrad = totalGrad + norm(val);
        end
    end

    totalGrad = totalGrad*decayFactor;


end

end