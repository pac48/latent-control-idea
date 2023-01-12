function plotAllCorrespondence(dlnet, imgInd)
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
        if imgInd > length(layer.h.structure.(layer.Name).mkptsNerf)
            continue
        end
        imRealBest = layer.h.structure.(layer.Name).imReal{imgInd};
        imgNerfBest = layer.h.structure.(layer.Name).imgNerf{imgInd};
        mkptsRealBest = layer.h.structure.(layer.Name).mkptsReal{imgInd};
        mkptsNerfBest = layer.h.structure.(layer.Name).mkptsNerf{imgInd};
        mconfBest = layer.h.structure.(layer.Name).mconf{imgInd};
        if isempty(mkptsNerfBest)
            continue
        end
        ind = ind+1;

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

