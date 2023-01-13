function [loss, gradients, state] = TFModelGradients(TFNet, Z, Tall, objects, allObjects)

loss = 0;
[map, state] = getNetOutput(TFNet, Z);

% [TsPred, state] = TFNet.predict(Z);
% TsPred = reshape(TsPred, 6, length(allObjects), []);

for i = 1:length(objects)
    object = objects{i};
    key = [object '_TF'];
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
    T = Tall{i};

    loss = loss + sum( (R-T(1:3,1:3,:)).^2 +  (P-T(1:3,end,:)).^2, 'all');
end

loss = loss;

gradients = dlgradient(loss, TFNet.Learnables);
loss = double(loss);
