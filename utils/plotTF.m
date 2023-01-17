function plotTF(T, style)
if isa(T, 'dlarray')
    T = extractdata(T);
end
point = T(1:3,end);
line =  [point point+T(1:3,1)*.15];
hold on
plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r', 'LineStyle', style)
line =  [point point+T(1:3,2)*.15];
hold on
plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g', 'LineStyle', style)
line =  [point point+T(1:3,3)*.15];
hold on
plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b', 'LineStyle', style)

end