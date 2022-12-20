classdef NerfLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        objNames
        nerf
        loftr
        h
        w
        fov
    end

    methods
        function layer = NerfLayer(nerf, loftr, objNames, h, w, fov)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = 'NerfLayer';
            % Set layer description.
            layer.Description = "Render image given camera xyz rpy and then extract 2D keypoints.";

            % Set layer type.
            layer.Type = "NerfLayer";
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'out1', 'out2'};

            layer.nerf = nerf;
            layer.loftr = loftr;
            layer.objNames = objNames;
            layer.h = h;
            layer.w = w;
            layer.fov = fov;

        end

        function [Z1, Z2] = predict(layer, X, Y)
            % X: xyz rpy (6 x batch) this is transform from world to camera
            % coordinates
            % Y: input image (h x w x 3 x b)
            X = double(gather(extractdata(X)));
            imReal = double(gather(extractdata(Y)));

            %             T = inv(reshape(X, 4, 4)); % cam to world
            T = reshape(X, 4, 4); % cam to world
            for i  = 1:length(layer.objNames)
                layer.nerf.setTransform({layer.objNames{i}, T})
                [img, depth] = layer.nerf.renderObject(layer.h, layer.w, layer.fov, layer.objNames{i});
            end

            % call LoFTR to get 2D points
            imgNerf = uint8(255*img);
            imReal = uint8(imReal);
            [mkptsReal, mkptsNerf, mconf] = layer.loftr.predict(imReal, imgNerf);
            if isempty(mkptsReal)
                warning('bad')
                Z1 = dlarray(ones(3,1, 1), 'SSB');
                Z2 = dlarray(ones(2,1,1),'SSB');
                return
            end

            %             mkptsReal(:,2) = layer.h - mkptsReal(:,2);
            %             mkptsNerf(:,2) = layer.h - mkptsNerf(:,2);

            inds = floor(mkptsNerf);
            inds = (inds(:,1,: )-1)*layer.h + inds(:,2,:);
            goodInds = depth(inds) ~= 0;
            inds = inds(goodInds);

            %             inds = (1:numel(depth))';
            d = depth(inds);

            [X,Y] = meshgrid(linspace(-.5, .5, layer.w), linspace(-.5,.5,layer.h));
            xPix = X(inds);
            yPix = -Y(inds);

            fl = 1;
            fx = fl*2*tand(layer.fov/2);
            fy = fx*layer.h/layer.w;

            xDir =  fx*xPix;
            yDir =  fy*yPix;
            zDir = -fl*ones(size(yDir));
            vec = [xDir yDir zDir];
            vec = vec./sqrt(sum(vec.^2, 2));
            points = vec.*d;
            points = points';  % camera coordinates

            %             Tinv = inv(T);
            points = pagemtimes(T(1:3, 1:3, :), points) + T(1:3, end, :); % world coordinates


            %             figure
            %             cData = permute(img, [3 1 2]);
            %             cData = reshape(cData, 3,[])';
            %             scatter3(points(1, :), points(2, :), points(3, :), 'CData', cData)
            %             axis equal

            %             close all
            %             imagesc(depth)
            % imshow(img)
            %             hold on
            %                 px = [mkptsNerf(:,1) mkptsNerf(:,1)];
            %                 py =  [mkptsNerf(:,2) mkptsNerf(:,2)];
            %                 plot(px', py', Marker='.');


            mkptsNerf = [mkptsNerf(goodInds,1) mkptsNerf(goodInds,2)];
            mkptsReal = [mkptsReal(goodInds,1) mkptsReal(goodInds,2)];
% 
            figure
            plotCorrespondence(imReal, imgNerf, mkptsReal, mkptsNerf, mconf)

            Z1 = dlarray(points, 'SSB');
            offset = [layer.w layer.h];
            mkptsRealNormalized = [fx -fy].*(mkptsReal - .5*offset)./offset;
            Z2 = dlarray(mkptsRealNormalized','SSB');

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
