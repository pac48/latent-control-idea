function [R, offset] = getTransform(model)
R = reshape(model([1 2 4 5]),2,2);
offset = -R*model([3 6])';
end

