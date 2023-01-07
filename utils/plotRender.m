function plotRender(dlnet)
layers = dlnet.Layers;
hold off
for layer = layers'
    if isa(layer, 'NerfLayer')
        if ~isfield(layer.h.structure, layer.Name)
            continue
        end
        imgNerfBest = layer.h.structure.(layer.Name).imgNerfBest;
        if ~isempty(imgNerfBest)

            if strcmp('nerf_background', layer.objNames{1})
                image(imgNerfBest)
            end
        end
    end
end
hold on

for layer = layers'
    if isa(layer, 'NerfLayer')
        if ~isfield(layer.h.structure, layer.Name)
            continue
        end
        imgNerfBest = layer.h.structure.(layer.Name).imgNerfBest;
        if ~isempty(imgNerfBest)
            if strcmp('nerf_cup', layer.objNames{1})
                background_ind = sum(imgNerfBest,3) ~= 0;
                image(imgNerfBest, 'AlphaData', background_ind)
            elseif strcmp('nerf_box', layer.objNames{1})
background_ind = sum(imgNerfBest,3) ~= 0;                
image(imgNerfBest, 'AlphaData', background_ind)
            end
        end

    end
end
end

