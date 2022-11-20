function data = customLoad(filename)
data = load(filename,'data_point');
data = {data.data_point};
end