rmdir("data/", 's')
mkdir("data/")

[X,Y] = meshgrid(-1:.03:1,-1:.03:1);

x_pos = 0;
y_pos = 0;

for s = 1:30000
    if rand < .5
img = 1*rand(size(X,1), size(X,2), 3);%+rand()*.6;
    else
img = 0*rand(size(X,1), size(X,2), 3)+rand(1,1,3)*.9;

    end
x_pos = (rand()-.5)*1.5;
y_pos = (rand()-.5)*1.5;

bool = (rand+.3)*(X-x_pos).^2 + (rand+.3)*(Y-y_pos).^2 < .2^2;
ind = randi(3)-1;
red = img(:,:,ind+1); 
green = img(:,:,mod(ind+1,3)+1);
blue = img(:,:,mod(ind+2,3)+1);

red(bool) = rand();
green(bool) = 0;
blue(bool) = 0;
img(:,:,ind+1) = red;
img(:,:,mod(ind+1,3)+1) = green;
img(:,:,mod(ind+2,3)+1) = blue;

% imshow(img)
% drawnow
data_point = struct('x', img, 'y', [x_pos, y_pos]);
save(['data/' num2str(s)], 'data_point')

s
end