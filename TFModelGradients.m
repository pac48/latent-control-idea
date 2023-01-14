function [loss, gradients, state] = TFModelGradients(FullTFNet, imgs, Z, Tall)
global curInd

vals = Tall.values;
assert(length(vals{1}) == size(imgs,4))

objects = getObjects(FullTFNet);

loss = 0;
for ind = 1:size(imgs,4)
    curInd = ind;
    [map, state] = getNetOutput(FullTFNet, imgs(:,:,:, ind), Z(:, ind));

    % [map, state] = getNetOutput(FullTFNet, imgs, Z);
    % [TsPred, state] = TFNet.predict(Z);
    % TsPred = reshape(TsPred, 6, length(allObjects), []);

    for i = 1:length(objects)
        object = objects{i};
        key = [object '_nerf_T_world_2_cam'];
        if ~map.isKey(key)
            continue
        end
        %         ind = find(contains(allObjects, object), 1);
        %     assert(~isempty(ind))
        %     TPred = TsPred(:, ind, :);
        %     TPred = reshape(TPred, 6, []);
        %     TPred = getT(TPred(1:3, :, :), TPred(4:6, :, :));
        TPred = map(key);
        TPred = reshape(TPred, 6, 1, []);

        R = getR(TPred(1:3, :, :));
        P = TPred(4:6, :, :);
        tmp = Tall(object);
        T = tmp{ind};
        if isempty(T)
            continue
        end

        loss = loss + sum( (R-T(1:3,1:3)).^2 +  (P-T(1:3,end)).^2, 'all');

    end
end

gradients = dlgradient(loss, FullTFNet.Learnables);
loss = double(loss);
