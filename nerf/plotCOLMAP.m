s = jsondecode(fileread('transforms.json'));
pAll = [];
for f = s.frames'
    p = f.transform_matrix(1:3, end);
    pAll = cat(2,pAll,p);
end

plot3(pAll(1,:), pAll(2,:), pAll(3,:), 'LineStyle','none', 'Marker','o')
axis equal
grid on


