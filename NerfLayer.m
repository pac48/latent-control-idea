classdef NerfLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct
    properties
        objNames
        nerf
        loftr
        height
        width
        fov
        numTransforms
    end

    methods
        function layer = NerfLayer(name, nerf, loftr, objNames, h, w, fov)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;%'NerfLayer';
            % Set layer description.
            layer.Description = "Render image given camera xyz rpy and then extract 2D keypoints.";

            % Set layer type.
            layer.Type = "NerfLayer";
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'out1', 'out2'};

            layer.nerf = nerf;
            layer.loftr = loftr;
            assert(length(objNames)==1, 'cannot handle multible objects')
            layer.objNames = objNames;
            layer.height = h;
            layer.width = w;
            layer.fov = fov;

            layer.h.structure.imReal = [];

            [allT, ~] = nerf.name2Frame(objNames{1});
            layer.numTransforms = length(allT);
            for j = 1:layer.numTransforms
                layer.h.structure.(['memoizedRender' layer.objNames{1} '_' num2str(j)]) = memoize(@layer.render);
            end
            if isfield(layer.h.structure, layer.Name)
                layer.h.structure = rmfield(layer.h.structure, layer.Name);
            end
            %             layer.h.structure.(['memoizedRender' layer.objNames{1}]).CacheSize = 1;

        end

        function clearCache(layer)
            for j = 1:layer.numTransforms
                layer.h.structure.(['memoizedRender' layer.objNames{1} '_' num2str(j)]).clearCache;
            end
        end

        function [fl, fx, fy] = getFValues(layer)
            fl = 1;
            fx = fl*2*tand(layer.fov/2);
            fy = fx*layer.height/layer.width;
        end

        function [mkptsNerf, mkptsReal, points, mconf, imgNerf] = render(layer, T)
            mkptsReal = [];
            points = [];
            mkptsNerf = [];
            imgNerf = [];
            mconf = [];
            if all(T==1,'all')
                return
            end

            T = inv(T); % cam to world

            %             T = reshape(X, 4, 4); % cam to world
            for i  = 1:length(layer.objNames)
                layer.nerf.setTransform({layer.objNames{i}, T})
                [img, depth] = layer.nerf.renderObject(layer.height, layer.width, layer.fov, layer.objNames{i});
            end

            % call LoFTR to get 2D points
            imgNerf = uint8(255*img);
            [mkptsReal, mkptsNerf, mconf] = layer.loftr.predict(layer.h.structure.imReal, imgNerf);

            if isempty(mkptsReal)
                return
            end

% Debug
%             if ~contains(layer.Name, 'background') && size(mkptsNerf, 1) >= 3
                maxDistance = 3;
                sampleSize = 3;
                [modelRANSAC, inlierIdx] = ransac2d(mkptsNerf, mkptsReal, sampleSize, maxDistance);
                mkptsNerf = mkptsNerf(inlierIdx, :);
                mkptsReal = mkptsReal(inlierIdx, :);
                mconf = mconf(inlierIdx);
%             end


            inds = floor(mkptsNerf);
            inds = (inds(:,1,: )-1)*layer.height + inds(:,2,:);
            goodInds = depth(inds) ~= 0;
            inds = inds(goodInds);
            if isempty(inds)
                return
            end
            
%             inds = (1:numel(depth))';
%             tmp = depth(inds) ~= 0;
%             inds = inds(tmp);

            d = depth(inds);

            [X,Y] = meshgrid(linspace(-.5, .5, layer.width), linspace(-.5, .5, layer.height));
            xPix = X(inds);
            yPix = -Y(inds);



            [fl, fx, fy] = layer.getFValues();

            xDir =  fx*xPix;
            yDir =  fy*yPix;
            zDir = -fl*ones(size(yDir));
            vec = [xDir yDir zDir];
            vec = vec./sqrt(sum(vec.^2, 2));
            points = vec.*d;
            points = points';  % camera coordinates
            points = pagemtimes(T(1:3, 1:3, :), points) + T(1:3, end, :); % world coordinates

% 
%                         figure
%                         imgR = img(:,:,1);
%                         imgG = img(:,:,2);
%                         imgB = img(:,:,3);
%                         imgR = imgR(inds);
%                         imgG = imgG(inds);
%                         imgB = imgB(inds);
%                         img = cat(3, imgR, imgG, imgB);
%                         cData = permute(img, [3 1 2]);
%                         cData = reshape(cData, 3, [])';
%                         scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)
%                         axis equal

            %                         close all
            %                         imagesc(depth)
            % imshow(img)
            %             hold on
            %                 px = [mkptsNerf(:,1) mkptsNerf(:,1)];
            %                 py =  [mkptsNerf(:,2) mkptsNerf(:,2)];
            %                 plot(px', py', Marker='.');


            mkptsNerf = [mkptsNerf(goodInds,1) mkptsNerf(goodInds,2)];
            mkptsReal = [mkptsReal(goodInds,1) mkptsReal(goodInds,2)];
            mconf = mconf(goodInds);
        end

        function [Z1, Z2] = predict(layer, Tin, Img)
            % X: xyz rpy (6 x batch) this is transform from world to camera
            % coordinates
            % Y: input image (h x w x 3 x b)
            Tin = double(gather(extractdata(Tin)));
            imReal = double(gather(extractdata(Img)));
            imReal = uint8(255*imReal);
            layer.h.structure.imReal = imReal;
            %             Z1 = dlarray(zeros(3, 1, 1), 'SSB');
            Z1 = dlarray(zeros(4, 1, 1), 'SSB');
            Z2 = dlarray(zeros(2, 1, 1),'SSB'); % real points

            mkptsRealBest = [];
            if isfield(layer.h.structure, layer.Name)
                jInds  = layer.h.structure.(layer.Name).jBest;
            else
                jInds = 1:size(Tin, 3);
            end

            for j = jInds
                T = Tin(:,:,j); % cam to world
                [mkptsNerf, mkptsReal, points, mconf, imgNerf] = layer.h.structure.(['memoizedRender' layer.objNames{1} '_' num2str(j)])(T);
                if length(jInds) == 1
                    mkptsReal = layer.h.structure.(layer.Name).mkptsRealBest;
                    mkptsNerf = layer.h.structure.(layer.Name).mkptsNerfBest;
                    points = layer.h.structure.(layer.Name).points;

%                     imgNerf = layer.h.structure.(layer.Name).imgNerfBest;
%                     mconf = layer.h.structure.(layer.Name).mconfBest;
                end
                %                 inds = find(mconf > .1, 9999);
                %                 mkptsNerf = mkptsNerf(inds, :);
                %                 mkptsReal =  mkptsReal(inds, :);
                %                 points = points(:,inds);

                if size(points, 2) > size(Z1, 2)
                    % hold off
                    % subplot(1,1,1)
                    % plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf, mconf)
                    % drawnow
                    layer.h.structure.(layer.Name).imRealBest = imReal;
                    layer.h.structure.(layer.Name).imgNerfBest = imgNerf;
                    layer.h.structure.(layer.Name).mkptsRealBest = mkptsReal;
                    layer.h.structure.(layer.Name).mkptsNerfBest = mkptsNerf;
                    layer.h.structure.(layer.Name).mconfBest = mconf;
                    layer.h.structure.(layer.Name).jBest = j;
                    layer.h.structure.(layer.Name).points = points;

                    points = cat(1, points, repmat(j, 1, size(points,2)));
                    Z1 = dlarray(points, 'SSB');
                    offset = [layer.width layer.height];
                    [fl, fx, fy] = layer.getFValues();
                    mkptsRealNormalized = [fx -fy].*(mkptsReal - .5*offset)./offset;
                    Z2 = dlarray(mkptsRealNormalized','SSB');
                end

                %                 if strcmp('nerf_cup', layer.objNames{1})
                %                     subplot(1,2,1)
                %                     imshow(imReal)
                %                     subplot(1,2,2)
                %                     imshow(imgNerf)
                %                     why
                %                 end
            end

            if isempty(size(Z1,2)==1)
                %                     subplot(1,2,1)
                %                     imshow(imReal)
                %                     subplot(1,2,2)
                %                     imshow(imgNerf)
                warning('bad')
            end


        end

        function [dLdX, dLdY] = backward(layer, X, Y, Z1, Z2, dLdZ1, dLdZ2, u3)
            % [dLdX, dLdAlpha] = backward(layer, X, ~, dLdZ, ~)
            % backward propagates the derivative of the loss function
            % through the layer.
            % Inputs:
            %         layer    - Layer to backward propagate through
            %         X        - Input data
            %         dLdZ     - Gradient propagated from the deeper layer
            % Outputs:
            %         dLdX     - Derivative of the loss with respect to the
            %                    input data
            %         dLdAlpha - Derivative of the loss with respect to the
            %                    learnable parameter Alpha
            %            error('not called normally')
            dLdX = X*0;
            dLdY = Y*0;
        end
    end
end
