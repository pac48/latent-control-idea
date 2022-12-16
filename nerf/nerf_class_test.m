close all
clear all
nerf = Nerf({'nerf_background', 'nerf_box', 'nerf_cup'});

%%
while 1
    tic
    [box, cup] = nerf.render('nerf_box','nerf_cup');
    subplot(1,2,1)
    imshow(box(:,:,4))
    subplot(1,2,2)
    imshow(cup(:,:,4))
    drawnow

    toc
end