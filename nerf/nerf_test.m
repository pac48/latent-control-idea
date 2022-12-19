% clear all;

% while 1
%     server = ZMQ_Server(5555, 100);
%     server = ZMQ_Server(5556, 100);
%     clear all;
% end
close all
clear all
% server1 = ZMQ_Server(5559, 100, 'nerf_box');
% server2 = ZMQ_Server(5561, 100, 'nerf_cup');
% server3 = ZMQ_Server(5563, 100, 'nerf_background');
% nerfBackground = Nerf('nerf_background');
% nerfBox = Nerf('nerf_box');
% nerfCup = Nerf('nerf_cup');

nerf = Nerf({'nerf_box', 'nerf_cup', 'nerf_background'});

%% background
% T3 = [0.2307445729283194, -0.32225770448710866, 0.918099620866566, 3.8365932351831096-3
%     0.9716467035084205, 0.026303540518213094, -0.2349698008514768, -0.7361525044292191
%     0.05157155806897006, 0.9462864764558558, 0.31919003546846786, 1.1337008735925902];
%
% T3 = [-0.5774626978959966,...
%     -0.07362735401276227,...
%     0.8130903057007832,...
%     0.9861726624436432,...
%     0.7065709810689659,...
%     0.45387712647637,...
%     0.5429115975490894,...
%     2.1498831732859793,...
%     -0.4090162359076458,...
%     0.8880172108363905,...
%     -0.21007415821001768,...
%     -0.5992285887575379...
%     0.0,...
%     0.0,...
%     0.0,...
%     1.0];

% p  = [.417; 1.369; 0.591];
% p  = p([2 3 1]);
% R = eye(3);
% 
% % T3 = cat(2, R, p);
% T3 = eye(4);
% T3(1:3, 1:3) = R;
% T3(1:3, end) = p;
% s = jsondecode(fileread('background.json'));
% T3 = s.transform_matrix;
allT = nerf.name2Frame('nerf_background');
% for i = 1:length(allT)
T3 = allT{14};

nerf.setTransform({'nerf_background', T3});
[backgroundImg, backgroundDepth]= nerf.renderObject(240, 320, 70, 'nerf_background');

subplot(1,2,1)
imshow(backgroundImg)
subplot(1,2,2)
imagesc(backgroundDepth)
clims = [min(backgroundDepth,[],'all') max(backgroundDepth,[],'all')];
clim(clims)
drawnow
% end
%% render
% v = VideoWriter('newfile.avi','Motion JPEG AVI');
% v.Quality = 95;
% open(v);
close all


tic
while 1
    y = 1*sin(toc*1);
    x = 1*cos(toc*1);
    T1 = [0.2307445729283194, -0.32225770448710866, 0.918099620866566, 3.8365932351831096-.5
        0.9716467035084205, 0.026303540518213094, -0.2349698008514768, -0.7361525044292191 + y
        0.05157155806897006, 0.9462864764558558, 0.31919003546846786, 1.1337008735925902 + 2
        0, 0, 0, 1];
    
    T1 = [[eul2rotm([pi/2, 0, pi/2+.2]); 0, 0, 0], [3.8; -0.7; -0.9; 1]];
    T1(1:3, end) = T1(1:3, end) + T1(1:3,3)*1.5;
    T1(1:3, end) = T1(1:3, end) + T1(1:3,1)*(y-2);

    %     T2 = [0.2307445729283194, -0.32225770448710866, 0.918099620866566, 3.8365932351831096-.5
    %         0.9716467035084205, 0.026303540518213094, -0.2349698008514768, -0.7361525044292191+x
    %         0.05157155806897006, 0.9462864764558558, 0.31919003546846786, 1.1337008735925902-.4];
%     T2 = [0, 0, 1, 3.5
%         1, 0, 0, 0
%         0, 1, 0, 0+1.2
%         0, 0, 0, 1];
%     
    
    T2 = [[eul2rotm([pi/2, 0, pi/2 + pi/12]); 0, 0, 0], [3.5; 0; -0.8; 1]];
    
    T2(1:3, end) = T2(1:3, end) + T2(1:3,3)*1.5;

    T2(1:3, end) = T2(1:3, end) + T2(1:3,2)*x;
    %     tmp = eye(4);
    %     tmp(1:3, 1:3) = eul2rotm([0, 0, toc]);
    %     T2Full = eye(4);
    %     T2Full(1:3, :) = T2;
    %     tmp = tmp*T2Full;
    %     T2 = tmp(1:3, :);

%     T1 = eye(4);
%     T2 = eye(4);
    nerf.setTransform({'nerf_box', T1});
    nerf.setTransform({'nerf_cup', T2});
    
    %     img1 = [];
    %     img2 = [];
    %     while isempty(img1) || isempty(img2)
    %         if isempty(img1)
    %             img1 = server1.recv();
    %         end
    %         if isempty(img2)
    %             img2 = server2.recv();
    %         end
    %     end
    fov = 70 + 10*sin(toc);
%     [backgroundImg, backgroundDepth]= nerf.renderObject(240, 320, fov, 'nerf_background');
    [boxImg, boxDepth, cupImg, cupDepth] = nerf.renderObject(240, 320, fov, 'nerf_box','nerf_cup');
%     [cupImg, cupDepth] = nerfCup.blockUntilResp();

%     [boxImg, box_background_ind] = removeBackground(boxImg);
%     boxDepth = boxDepth.*box_background_ind;
%     [cupImg, cup_background_ind] = removeBackground(cupImg);
%     cupDepth = cupDepth.*cup_background_ind;

    %     img = boxImg + cupImg;
    %     depth = cupDepth + boxDepth;

    %     img(img==0) = img3(img==0);
    %     img = lin2rgb(img);
    %     low = 0.1;
    %     high = 1.0;
    %     img = imadjust(img,[low high],[]); % I is double
%     boxImg = colorAdjust(boxImg);
%     cupImg = colorAdjust(cupImg);

    subplot(1,2,1)
    hold off
    image(backgroundImg);
    hold on
    box_background_ind = boxDepth ~= 0;
    image(boxImg, 'AlphaData', box_background_ind)
    cup_background_ind = cupDepth ~= 0;
    image(cupImg, 'AlphaData', cup_background_ind)
    axis equal

    subplot(1,2,2)
    hold off
    image(backgroundDepth);
    hold on
    imagesc(boxDepth, 'AlphaData', box_background_ind)
    imagesc(cupDepth, 'AlphaData', cup_background_ind)
    axis equal
    clims = [0 5];
%     clims = [min(backgroundDepth,[],'all') max(backgroundDepth,[],'all')];
    clim(clims)


    drawnow

    %   frame = getframe(gcf);
    %    writeVideo(v,frame);

end
% close(v);


function img = colorAdjust(img)
img = lin2rgb(img);
low = 0.1;
high = 1.0;
img = imadjust(img,[low high],[]); % I is double

end

function [img, background_ind] = removeBackground(img)
img = img(:,:,1:size(img,3));
imgBlur = imgaussfilt(img,.5);
background_ind = any(imgBlur > .04, 3);
img = img.*background_ind;
end

