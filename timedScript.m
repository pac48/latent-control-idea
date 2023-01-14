initialLearnRateTF = 1e-1;
decayTF = 0.000005;
momentumTF = 0.95;
velocityTF = [];

iteration = 0;

averageGrad = [];
averageSqGrad = [];

setDetectObjects(FullTFNet, {})

while 1
    tic

    [lossTF, gradients, state] = dlfeval(@TFModelGradients, FullTFNet, imgs, Z + .1*(rand(size(Z))-.5), objects, Tall);
    lossTF
    learnRateTF = initialLearnRateTF/(1 + decayTF*iteration);
    [FullTFNet, velocityTF] = sgdmupdate(FullTFNet, gradients, velocityTF, learnRateTF, momentumTF);

    iteration = iteration + 1;
    toc

end


%%
cam = webcam()

while 1
img = cam.snapshot();
image(img)
% pause(2)

end

%% sanity
close all

load('debug1.mat')
imReal1 = imReal;
imgNerfRot1 = imgNerfRot;

[mkptsReal, mkptsNerf, mconf] = loftr.predict(imReal, imgNerfRot)
figure; imshow(imReal)
figure; imshow(imgNerfRot)


load('debug2.mat')
imReal2 = imReal;
imgNerfRot2 = imgNerfRot;
[mkptsReal, mkptsNerf, mconf] = loftr.predict(imReal, imgNerfRot)
figure; imshow(imReal)
figure; imshow(imgNerfRot)

all(imReal1 == imReal2, 'all')
all(imgNerfRot1 == imgNerfRot2, 'all')
