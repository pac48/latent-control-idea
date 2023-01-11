function [R, offset] = getTransform(model)
% R = eye(2);
% R(1,1) = model(1);
% R(1,2) = model(2);
% R(2,1) = model(4);
% R(2,2) = model(5);
% R = reshape(model([1 2 4 5]), 2, 2)';
% offset = model([3 6])';

R = model(1:2, 1:2);
offset = model(1:2,end);
end

