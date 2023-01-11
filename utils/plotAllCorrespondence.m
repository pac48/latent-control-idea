function plotAllCorrespondence(dlnet)
layers = dlnet.Layers;
count = 0;
for layer = layers'
    if isa(layer, 'NerfLayer')
        count = count+1;
    end
end
ind = 0;
for layer = layers'
    if isa(layer, 'NerfLayer')
        ind = ind+1;
        imRealBest = layer.h.structure.(layer.Name).imRealBest;
        imgNerfBest = layer.h.structure.(layer.Name).imgNerfBest;
        mkptsRealBest = layer.h.structure.(layer.Name).mkptsRealBest;
        mkptsNerfBest = layer.h.structure.(layer.Name).mkptsNerfBest;
        mconfBest = layer.h.structure.(layer.Name).mconfBest;
        jBest = layer.h.structure.(layer.Name);
        if isempty(mkptsNerfBest)
            continue
        end

        %         if ~isempty(mkptsRealBest)
        %             hold off
        %             if strcmp('nerf_background', layer.objNames{1})
        %                 subplot(1,3,1)
        %             elseif strcmp('nerf_cup', layer.objNames{1})
        %                 subplot(1,3,2)
        %             elseif strcmp('nerf_box', layer.objNames{1})
        %                 subplot(1,3,3)
        %             end
        %         end
        hold off
        subplot(1, count, ind)
        plotCorrespondence(imRealBest, imgNerfBest, mkptsRealBest, mkptsNerfBest, mconfBest)
        drawnow
    end
end
end

