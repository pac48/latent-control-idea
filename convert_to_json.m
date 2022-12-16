% sub = rossubscriber('/camera/color/camera_info');
% msg = sub.LatestMessage;
tmp = load("camera_info_msg.mat");
msg = tmp.msg;
tmp = load("calibrate/T.mat");
Toffset = tmp.T;
% msg.Height = 1080;
% msg.Width = 1920;
% camera_angle_x  = pi*69/180;
% camera_angle_y = pi*42/180;

camera_angle_x = 2 * atan(double(msg.width)/(2*msg.k(1) ));
% camera_angle_x = 1.500222222;
camera_angle_y = 2 * atan(double(msg.height)/(2 * msg.k(5)));
% camera_angle_y = 0.994333333;
% 
config_struct = struct( ...
    "camera_angle_x", camera_angle_x, ...
    "camera_angle_y", camera_angle_y, ...
    "fl_x", msg.k(1), ...
    "fl_y", msg.k(5), ...
    "k1", msg.d(1), ...
    "k2", msg.d(2), ...
    "p1", msg.d(3), ...
    "p2", msg.d(4), ...
    "cx", msg.k(3), ...
    "cy", msg.k(6), ...
    "w", msg.width, ...
    "h", msg.height, ...
    "aabb_scale", 16, ... % set size of unit cube used for nerf, e.g side length is 4
    "frames", []);


% 
%   config_struct = struct( "camera_angle_x", 0.9634420875003629,...
%   "camera_angle_y", 0.7519462488072212,...
%   "fl_x", 612.0883828175882,...
%   "fl_y", 607.9783338750312,...
%   "k1", 0.14945910491933814,...
%   "k2", -0.30731986401027184,...
%   "p1", -0.0033799127880072893,...
%   "p2", 0.0025187737868227525,...
%   "cx", 319.3435446484628,...
%   "cy", 241.9909904159064,...
%   "w", 640.0,...
%   "h", 480.0,...
%   "aabb_scale", 16,...
%   "frames", []);


files = dir('data');

count = 0;
pos = zeros(3,1);
for file = files'
    if file.bytes>0
        s = load(fullfile(file.folder, file.name) );
        pos = pos + s.dataPoint.T(1:3,end);
        count = count+1;
    end
end
pos = pos./count;
pos(3) = pos(3) - .3;
% pos = pos*0;

close all

delete(fullfile('box', 'images', '*'))
ind = 0;
for file = files'
    if file.bytes>0
%         ind = ind + 1;
        tmp = split(file.name,'_');
        tmp = tmp{2};
        tmp = split(tmp,'.');
        ind = tmp{1};
        s = load(fullfile(file.folder, file.name) );
        img = s.dataPoint.img;
        %         img= 255-img;
        img_path = fullfile('images', [ind '.jpg'] );
        imwrite(img, fullfile('box', img_path), 'jpg');
        % y is up, z is forward, but everything is negated
        T = s.dataPoint.T;
        T = T*Toffset;
        %         T(1:3,1:3) = [0 1 0;
        %                       0 0 -1
        %                       1 0 0]*T(1:3,1:3);

        xAxis = T(1:3,1);
        yAxis = T(1:3,2);
        zAxis = T(1:3,3);

        %         T(1:3,3) = xAxis;
        %         T(1:3,2) = -yAxis;
        %         T(1:3,1) = zAxis;

% good one
%                 T(1:3,3) = -xAxis;
%                 T(1:3,2) = zAxis;
%                 T(1:3,1) = -yAxis;

%         T(1:3,3) = xAxis;
%         T(1:3,2) = -zAxis;
%         T(1:3,1) = cross(T(1:3,2), T(1:3,3));
%         T(1:3,1) =  T(1:3,1)./norm( T(1:3,1));


        T(1:3,end) = (T(1:3,end) - pos)*10;

        tmp = struct("file_path", img_path, "sharpness", 20, "transform_matrix", []);
        tmp.transform_matrix = T;

        config_struct.frames = cat(1,config_struct.frames,tmp);


        point = T(1:3,end);
        line =  [point point+T(1:3,1)*.05];
        hold on
        plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','r')
        line =  [point point+T(1:3,2)*.05];
        hold on
        plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','g')
        line =  [point point+T(1:3,3)*.05];
        hold on
        plot3(line(1,:), line(2,:), line(3,:),'MarkerSize',10, 'Marker','.', 'Color','b')

    end
end
axis equal

str = jsonencode(config_struct);
str  = strrep(str , ',',',\n');

fileName = 'transforms.json';
fid = fopen(fullfile('box', fileName), 'w');
fprintf(fid, str);
fclose(fid);

%% valdiate
s = jsondecode(fileread(fullfile('box', fileName)));

tmp = arrayfun(@(x) x.file_path,  s.frames,  'UniformOutput',  false);
tmp = cellfun(@(x) split(x,'/'),  tmp,  'UniformOutput',  false);
tmp = cellfun(@(x) split(x{end},'.'),  tmp,  'UniformOutput',  false);
tmp = cellfun(@(x) str2num(x{1}),  tmp,  'UniformOutput',  true);

[~, idx] = sort(tmp);
frames = s.frames(idx);

p = [-2.58866 2.55508 2.00647]';
tmp = arrayfun(@(x) norm(x.transform_matrix(1:3,end)-p), frames );
find(min(tmp)==tmp)

hold off
for f = 1:length(frames)-1
    frame1 = frames(f);
    frame2 = frames(f+1);
    p1 = frame1.transform_matrix(1:3,end);
    p2 = frame2.transform_matrix(1:3,end);
    plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)] );hold on

end

