function img = plotRender(dlnet, imgInd)
layers = dlnet.Layers;
hold off
for layer = layers'
    if isa(layer, 'NerfLayer') && strcmp('nerf_background', layer.objNames{1})
        if ~isfield(layer.h.structure, layer.Name)  || imgInd > length(layer.h.structure.(layer.Name).imgNerf)
            continue
        end
        imgNerf = layer.h.structure.(layer.Name).imgNerf{imgInd};
        if ~isempty(imgNerf)
                image(imgNerf)
                img = imgNerf;
        end
    end
end
hold on


for layer = layers'
    if isa(layer, 'NerfLayer')
        if ~isfield(layer.h.structure, layer.Name) || imgInd > length(layer.h.structure.(layer.Name).imgNerf)
            continue
        end
        imgNerf = layer.h.structure.(layer.Name).imgNerf{imgInd};
        if ~isempty(imgNerf)
%             if strcmp('nerf_cup', layer.objNames{1})
                background_ind = sum(imgNerf,3) ~= 0;
                image(imgNerf, 'AlphaData', background_ind)
                set(gca, 'YDir','reverse')

                img = img.*uint8(1-background_ind) + imgNerf.*uint8(background_ind);

%             elseif strcmp('nerf_box', layer.objNames{1})
%                 background_ind = sum(imgNerf,3) ~= 0;                
%                 image(imgNerf, 'AlphaData', background_ind)
%             end
        end

    end
end
end

