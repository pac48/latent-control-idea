function plotAllCorrespondence(dlnet)
layers = dlnet.Layers;
for layer = layers'
    if isa(layer, 'NerfLayer')
        if ~isfield(layer.h.structure, layer.Name)
            continue
        end
        imRealBest = layer.h.structure.(layer.Name).imRealBest;
        imgNerfBest = layer.h.structure.(layer.Name).imgNerfBest;
        mkptsRealBest = layer.h.structure.(layer.Name).mkptsRealBest;
        mkptsNerfBest = layer.h.structure.(layer.Name).mkptsNerfBest;
        mconfBest = layer.h.structure.(layer.Name).mconfBest;
        jBest = layer.h.structure.(layer.Name);

        if ~isempty(mkptsRealBest)
            hold off
            if strcmp('nerf_background', layer.objNames{1})
                subplot(1,3,1)
            elseif strcmp('nerf_cup', layer.objNames{1})
                subplot(1,3,2)
            elseif strcmp('nerf_box', layer.objNames{1})
                subplot(1,3,3)
            end
        end
        plotCorrespondence(imRealBest, imgNerfBest, mkptsRealBest, mkptsNerfBest, mconfBest)
        drawnow
    end
end
end

