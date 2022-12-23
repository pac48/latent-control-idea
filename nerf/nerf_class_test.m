close all
clear all
nerf = Nerf({'nerf_background', 'nerf_box', 'nerf_cup'});
% nerf = Nerf({'nerf_box'});
% nerf = Nerf({'nerf_box', 'nerf_cup'});

%%
for i =1:500
    tic
    out = nerf.renderObject(640/2, 480/2, 70, 'nerf_background', 'nerf_box','nerf_cup', 'nerf_box2');

%         box = nerf.renderObject(640/2, 480/2, 70, 'nerf_box');
%         cup = nerf.renderObject(640/2, 480/2, 70, 'nerf_cup');
%         back = nerf.renderObject(640/2, 480/2, 70, 'nerf_background');
%         % 0.071743
    
        toc
    %     subplot(1,2,1)
    %     imshow(box(:,:,1:3))
    %     subplot(1,2,2)
    %     imshow(cup(:,:,1:3))
    %     drawnow


end