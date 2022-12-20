function img = readFunc(fileName)
img = imread(fileName);
img = imresize(img, [240, 320]);
img = double(img)./255;
img = dlarray(img, 'SSC');
end