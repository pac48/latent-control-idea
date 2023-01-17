function mapBatch = mergeMaps(maps)
mapBatch  = containers.Map();
for i = 1:length(maps)
    map = maps{i};
    map_keys = map.keys;
    for k = 1:length(map_keys)
        key = map_keys{k};
        if ~isKey(mapBatch, key)
            mapBatch(key) = {};
        end
        val = map(key);
        mapBatch(key) = cat(2, mapBatch(key), {val});
    end

end

end