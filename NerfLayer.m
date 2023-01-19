classdef NerfLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct
    properties
        objNames
        height
        width
        %         fov
        s
        objectName
        initNerfImgs
        initNerfDepths
        allT
    end
    methods(Static)
        function llayer = loadobj(layer)
            layer.h.structure.detectObjects = {};
            llayer = layer;
        end
    end
    methods
        function layer = NerfLayer(name, allT, objNames, h, w)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name;%'NerfLayer';
            tmp = split(name, '_');
            layer.objectName = strjoin(tmp(1:end-2), '_');
            % Set layer description.
            layer.Description = "Render image given camera xyz rpy and then extract 2D keypoints.";

            % Set layer type.
            layer.Type = "NerfLayer";
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'pointsNerf', 'mkptsReal', 'mkptsNerf', 'imgReal', 'imgNerf', 'depth'};

            assert(length(objNames)==1, 'cannot handle multible objects')
            layer.objNames = objNames;
            layer.height = h;
            layer.width = w;
            %             layer.fov = fov;
            layer.allT = allT;

            layer.s = RandStream('mlfg6331_64');

            layer.h.structure.imReal = [];
            layer.h.structure.detectObjects = {};

            [layer.initNerfImgs, layer.initNerfDepths] = layer.renderInit();

            layer.h.structure.(layer.Name).skipNerf = true;

            layer.clearCache()


        end

        function clearCache(layer)
            layer.h.structure.(layer.Name).imReal = {};
            layer.h.structure.(layer.Name).imgNerf = {};
            layer.h.structure.(layer.Name).imgNerf = {};
            layer.h.structure.(layer.Name).depthNerf = {};
            layer.h.structure.(layer.Name).mkptsReal = {};
            layer.h.structure.(layer.Name).mkptsNerf = {};
            layer.h.structure.(layer.Name).mconf = {};
            layer.h.structure.(layer.Name).points = {};
            layer.h.structure.(layer.Name).indT = {};
            layer.h.structure.(layer.Name).angle = {};
        end

        function [fl, fx, fy] = getFValues(layer)
            global fov
            fovL = fov;

            fl = 1;
            fx = fl*2*tand(fovL/2);
            fy = fx*layer.height/layer.width;
        end

        function [imgs, depths] = renderInit(layer)
            global nerf
            global fov

            imgs = cell(1, length(layer.allT));
            depths = cell(1, length(layer.allT));
            for i = 1:length(layer.allT)
                T = layer.allT{i};
                nerf.setTransform({layer.objNames{1}, T})
                [imgNerf, depth] = nerf.renderObject(layer.height, layer.width, fov, layer.objNames{1});
                imgs{i} = imgNerf;
                depths{i} = depth;
            end
        end

        function inds = findInitMatch(layer, imReal)
            global loftr

            batchSize = size(imReal,4);
            inds = cell(1, batchSize);
            foundAngles = cell(1, batchSize);

            for b = 1:batchSize
                numPoint = 2;
                tmp = [0:30:170; -(0:30:170)];
                tmp = reshape(tmp, 1, []);
                tmp(diff(tmp)==0) = [];
                for angle = 0 %tmp
                    for i = 1:length(layer.initNerfImgs)
                        T = layer.allT{i};
                        imgNerf = layer.initNerfImgs{i};
                        depth = layer.initNerfDepths{i};
                        [mkptsNerf, mkptsReal, points, mconf] = layer.matchImage(imReal(:,:,:,b), imgNerf, depth, T, angle);

                        if size(points, 2) > numPoint
                            numPoint = size(points, 2);
                            inds{b} = i;
                            foundAngles{b} = angle;
                            layer.updateState(b, mkptsNerf, mkptsReal, points, mconf, imgNerf, imReal, depth)
                        end
                    end
                    if numPoint ~=2
                        break;
                    end
                end
            end

            layer.h.structure.(layer.Name).indT = inds;
            layer.h.structure.(layer.Name).angle = foundAngles;

        end

        function [mkptsNerf, mkptsReal, points, mconf, imgNerf, depth] = render(layer, T, angle, imReal)
            global nerf
            global fov

            T = inv(T); % cam to world

            assert(length(layer.objNames)==1, "only one object is supported per nerf")

            nerf.setTransform({layer.objNames{1}, T})
            [img, depth] = nerf.renderObject(layer.height, layer.width, fov, layer.objNames{1});

            imgNerf = uint8(255*img);
            [mkptsNerf, mkptsReal, points, mconf] =  layer.matchImage(imReal, imgNerf, depth, T, angle);

        end

        function [mkptsNerf, mkptsReal, points, mconf] = matchImage(layer, imReal, imgNerf, depth, T, angle)
            global loftr
            points = [];

            if  isa(imReal, 'dlarray')
                imReal = gather(extractdata(imReal));
            end
            if  isa(imReal, 'double')
                imReal = uint8(255*imReal);
            end
            if  isa(imgNerf, 'double')
                imgNerf = uint8(255*imgNerf);
            end

            A = [cosd(angle) -sind(angle) 0; sind(angle) cosd(angle) 0; 0 0 1];

            tform = affinetform2d(A);
            centerOutput = affineOutputView(size(imgNerf), tform, "BoundsStyle","CenterOutput");
            imgNerfRot = imwarp(imgNerf, tform, "OutputView",centerOutput);

            % call LoFTR to get 2D points
            [mkptsReal, mkptsNerf, mconf] = loftr.predict(imReal, imgNerfRot);

            if size(mkptsNerf,1) < 3
                return
            end


            tmp = inv(A)*cat(2, mkptsNerf - [layer.width/2 layer.height/2], ones(size(mkptsNerf, 1), 1) )';
            mkptsNerf = tmp(1:2,:)' + [layer.width/2 layer.height/2];

            inds = floor(mkptsNerf);
            inds = (inds(:,1,: )-1)*layer.height + inds(:,2,:);
            if any(inds <= 0) || any(inds > numel(depth))
                return;
            end

            %             goodInds = depth(inds) ~= 0;
            goodInds = depth(inds) > 1;

            inds = inds(goodInds);
            if length(inds) < 3
                return
            end

            %             if false
            %                 inds = (1:numel(depth))';
            %                 %                         tmp = depth(inds) ~= 0;
            %                 %                         inds = inds(tmp);
            %                 d = depth(inds)*0+1;
            %                 [X,Y] = meshgrid(linspace(-.5, .5, layer.width), linspace(-.5, .5, layer.height));
            %                 xPix = X(inds);
            %                 yPix = -Y(inds);
            %                 [fl, fx, fy] = layer.getFValues();
            %                 xDir =  fx*xPix;
            %                 yDir =  fy*yPix;
            %                 zDir = -fl*ones(size(yDir));
            %                 vec = [xDir yDir zDir];
            %                 vec = vec./sqrt(sum(vec.^2, 2));
            %                 points = vec.*d;
            %                 points = points';  % camera coordinates
            %                 %             points = pagemtimes(T(1:3, 1:3, :), points) + T(1:3, end, :); % world coordinates
            %                 figure
            %                 imgR = img(:,:,1);
            %                 imgG = img(:,:,2);
            %                 imgB = img(:,:,3);
            %                 imgR = imgR(inds);
            %                 imgG = imgG(inds);
            %                 imgB = imgB(inds);
            %                 imgtmp = cat(3, imgR, imgG, imgB);
            %                 cData = permute(imgtmp, [3 1 2]);
            %                 cData = reshape(cData, 3, [])';
            %                 scatter3(points(1, :), points(2, :), points(3, :)*0+1, 'CData', cData)
            %                 axis equal
            %             end

            d = depth(inds);

            [X,Y] = meshgrid(linspace(-.5, .5, layer.width), linspace(-.5, .5, layer.height));
            xPix = X(inds);
            yPix = -Y(inds);

            [fl, fx, fy] = layer.getFValues();

%             xDir =  fx*xPix;
%             yDir =  fy*yPix;
%             zDir = -fl*ones(size(yDir));
%             vec = [xDir yDir zDir];
%             vec = vec./sqrt(sum(vec.^2, 2));
%             points = vec.*d;

            Z = -d;
            X = -(xPix).*Z./(1/fx);
            Y = -(yPix).*Z./(1/fy);
            points = [X Y Z];
            
            points = points';  % camera coordinates
            points = pagemtimes(T(1:3, 1:3, :), points) + T(1:3, end, :); % world coordinates

            mkptsNerf = [mkptsNerf(goodInds,1) mkptsNerf(goodInds,2)];
            mkptsReal = [mkptsReal(goodInds,1) mkptsReal(goodInds,2)];
            mconf = mconf(goodInds);

            if ~contains(layer.Name, 'background')
                maxDistance = 20;
                inds = getFilteredInds(maxDistance, imReal, imgNerf, mkptsReal, mkptsNerf);

                mkptsNerf = mkptsNerf(inds, :);
                mkptsReal = mkptsReal(inds, :);
                mconf = mconf(inds);
                points = points(:, inds);

            else
                inds = [];
                for i = 1:size(mkptsReal, 1)
                    idx = floor(mkptsReal(i, :));
                    c1 = double(imReal(idx(2), idx(1), :))./255;
                    
                    idx = floor(mkptsNerf(i, :));
                    c2 = double(imgNerf(idx(2), idx(1), :))./255;
                    if sum(abs(c1-c2)) < .1
                        inds = cat(1, inds, i);
                    end
                
                end
                
                mkptsNerf = mkptsNerf(inds, :);
                mkptsReal = mkptsReal(inds, :);
                mconf = mconf(inds);
                points = points(:, inds);

            end
            %


            if length(mconf) < 3
                mkptsNerf = [];
                mkptsReal = [];
                mconf  = [];
                points  = [];
                return
            end



            if contains(layer.Name, 'background')
                model = fitModel([mkptsNerf mkptsReal]);
                cost = fitValue(model, [mkptsNerf mkptsReal]);

%                 percentiles = prctile(cost, 95);
                percentiles = prctile(cost, 99);
                inds = 1:length(mconf);
                inds = inds(cost <= percentiles(1));
%                 inds = inds(cost <= 150);
                mkptsNerf = mkptsNerf(inds, :);
                mkptsReal = mkptsReal(inds, :);
                mconf = mconf(inds);
                points = points(:, inds);
                cost = cost(inds);
            else
                model = fitModel([mkptsNerf mkptsReal]);
                cost = fitValue(model, [mkptsNerf mkptsReal]);
                %                 for j =1:5
                %                     model = fitModel([mkptsNerf mkptsReal]);
                %                     cost = fitValue(model, [mkptsNerf mkptsReal]);
                %
                %                     percentiles = prctile(cost, 90);
                %                     inds = 1:length(mconf);
                %                     inds = inds(cost <= percentiles(1));
                %                     mkptsNerf = mkptsNerf(inds, :);
                %                     mkptsReal = mkptsReal(inds, :);
                %                     mconf = mconf(inds);
                %                     points = points(:, inds);
                %                     cost = cost(inds);
                %                 end
            end


                        if mean(cost) > 200
                            mkptsNerf = [];
                            mkptsReal = [];
                            mconf  = [];
                            points  = [];
                            return
                        end


            %
            %             if contains(layer.Name, 'block')
            %                 why
            %             end


            %
            %             maxDistance = 100;
            %             sampleSize = 3;
            %             [modelRANSAC, inlierIdx] = ransac2d(mkptsNerf, mkptsReal, sampleSize, maxDistance);
            %             tmp = evaluateModel(modelRANSAC, mkptsNerf)
            %             inds = 1:size(mkptsReal,1);
            %             inds = inds(inlierIdx);
            %             mkptsNerf = mkptsNerf(inds, :);
            %             mkptsReal = mkptsReal(inds, :);
            %             mconf = mconf(inds);
            %             points = points(:, inds);
            %             cost = fitValue(modelRANSAC, [mkptsNerf mkptsReal]);
        end

        function updateState(layer, curInd, mkptsNerf, mkptsReal, points, mconf, imgNerf, imReal, depth)
            layer.h.structure.(layer.Name).imgNerf{curInd} = imgNerf;
            layer.h.structure.(layer.Name).imReal{curInd} = imReal;
            layer.h.structure.(layer.Name).depth{curInd} = depth;
            if length(layer.h.structure.(layer.Name).points) < curInd
                layer.h.structure.(layer.Name).points{curInd} = [];
            end

            if ~contains(layer.Name, 'background') && ~isempty(points) && size(points, 2) >=  5 %size(layer.h.structure.(layer.Name).points{curInd}, 2)%3 %size(layer.h.structure.(layer.Name).points, 2)
                layer.h.structure.(layer.Name).depthNerf{curInd} = [];
                layer.h.structure.(layer.Name).mkptsReal{curInd} = mkptsReal;
                layer.h.structure.(layer.Name).mkptsNerf{curInd} = mkptsNerf;
                layer.h.structure.(layer.Name).mconf{curInd} = mconf;
                layer.h.structure.(layer.Name).points{curInd} = points;
            end
            if contains(layer.Name, 'background') && ~isempty(points) && size(points, 2) >= 100 %size(layer.h.structure.(layer.Name).points{curInd}, 2)
                layer.h.structure.(layer.Name).depthNerf{curInd} = [];
                layer.h.structure.(layer.Name).mkptsReal{curInd} = mkptsReal;
                layer.h.structure.(layer.Name).mkptsNerf{curInd} = mkptsNerf;
                layer.h.structure.(layer.Name).mconf{curInd} = mconf;
                layer.h.structure.(layer.Name).points{curInd} = points;
            end


        end

        function [points, mkptsRealNormalized, mkptsNerf, imgReal, imgNerf, depth] = predict(layer, Tin, Img)
            % X: xyz rpy (6 x batch) this is transform from world to camera
            % coordinates
            % Y: input image (h x w x 3 x b)
            global curInd

            Tin = double(gather(extractdata(Tin)));
            imReal = double(gather(extractdata(Img)));

            %             'pointsNerf', 'mkptsReal', 'mkptsNerf', 'imgReal', 'imgNerf', 'depth'
            points = dlarray([], 'SSB'); % pointsNerf
            mkptsRealNormalized = dlarray([], 'SSB'); % real points 2d : mkptsReal
            mkptsNerf = dlarray([], 'SSB'); % mkptsNerf

            imgReal = dlarray([], 'SSCB'); % imgReal
            imgNerf = dlarray([], 'SSCB'); % imgNerf
            depth = dlarray([], 'SSCB'); % depth

            if ~any(cellfun(@(x) strcmp(x, layer.objectName),  layer.h.structure.detectObjects))
                return
            end
            if all(Tin(:,:,1)==1, 'all')
                return
            end

            if all(Tin(:,:,1)==1, 'all')
                return
            end

            if size(Tin, 1) == 6 % convert to T
                Tin = reshape(Tin, 6,1,[]);
                Tin = getT(Tin(1:3,:), Tin(4:6,:));
            else
                Tin = reshape(Tin, 4, 4, []);
            end

            %             curInd = layer.h.structure.curInd;
            %             inds = layer.h.structure.(layer.Name).indT{curInd};
            %             angle = layer.h.structure.(layer.Name).angle{curInd};
            angle = 0;

            skipNerf = layer.h.structure.(layer.Name).skipNerf;
            if ~skipNerf %&& ~isempty(inds)
                %                 for b =  1:size(Tin, 3)
                %                     if isempty(inds)
                %                         break;
                %                     end
                T = Tin;%(:,:,b); % world to cam
                [mkptsNerf, mkptsReal, points, mconf, imgNerf, depth] = layer.render(T, angle, imReal);%layer.h.structure.(['memoizedRender' layer.objNames{1} '_' num2str(j)])(T);
                layer.updateState(curInd, mkptsNerf, mkptsReal, points, mconf, imgNerf, imReal, depth);
                %                     layer.h.structure.(layer.Name).imgNerf{curInd} = imgNerf;
                %                     layer.h.structure.(layer.Name).imReal{curInd} = imReal;
                %                     if ~isempty(points) && size(points, 2) >=  3 %size(layer.h.structure.(layer.Name).points, 2)
                %                         layer.h.structure.(layer.Name).depthNerf{curInd} = [];
                %                         layer.h.structure.(layer.Name).mkptsReal{curInd} = mkptsReal;
                %                         layer.h.structure.(layer.Name).mkptsNerf{curInd} = mkptsNerf;
                %                         layer.h.structure.(layer.Name).mconf{curInd} = mconf;
                %                         layer.h.structure.(layer.Name).points{curInd} = points;
                %                     end
                %                 end

                %                 if ~isempty(points) % points found
                %                     layer.h.structure.(layer.Name).rotAngle = angle;
                %                     break
                %                 end
                %                 if ~isempty(layer.h.structure.(layer.Name).points) % previous points found
                %                     break
                %                 end
                %             end
                %         end
            end

            if curInd > length(layer.h.structure.(layer.Name).points) || isempty(layer.h.structure.(layer.Name).points{curInd})
                %                     subplot(1,2,1)
                %                     imshow(imReal)
                %                     subplot(1,2,2)
                %                     imshow(imgNerf)
                %                 warning('bad bad')
            else
                tmp = layer.h.structure.(layer.Name);
                mkptsReal = tmp.mkptsReal{curInd};
                points = tmp.points{curInd};

                points = dlarray(points, 'SSB');
                offset = [layer.width layer.height];
                [fl, fx, fy] = layer.getFValues();
                mkptsRealNormalized = [fx -fy].*(mkptsReal - .5*offset)./offset;
                mkptsRealNormalized = dlarray(mkptsRealNormalized','SSB');

                if ~skipNerf
                    mkptsNerf = tmp.mkptsNerf{curInd};
                    imgNerf = tmp.imgNerf{curInd};
                    depth = tmp.depth{curInd};

                    mkptsNerf = dlarray(mkptsNerf', 'SSB'); % mkptsNerf
                    imgReal = dlarray(imReal, 'SSCB'); % imgReal
                    imgNerf = dlarray(double(imgNerf)./255, 'SSCB'); % imgNerf
                    depth = dlarray(depth, 'SSCB'); % depth
                end

            end


        end

        function [dLdX, dLdY] = backward(layer, X, Y, Z1, Z2, Z3, Z4, Z5, Z6, dLdZ1, dLdZ2, dLdZ3, dLdZ4, dLdZ5, dLdZ6, u3)
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
            %             dLdW = W*0;
            dLdX = X*0;
            dLdY = Y*0;
        end
    end
end
