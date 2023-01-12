classdef TFOffsetLayer < nnet.layer.Layer & nnet.layer.Formattable & GlobalStruct
    properties
        T0
        allT
        objectName
    end
    methods(Static)
        function llayer = loadobj(layer)
            llayer = TFOffsetLayer(layer.Name, layer.allT);
        end
    end
    methods
        function layer = TFOffsetLayer(name, allT)
            % Create a TFLayer.

            % Set layer name.
            layer.Name = name; %'TFOffsetLayer';
            tmp = split(name, '_');
            layer.objectName = strjoin(tmp(1:end-2), '_');
            layer.allT = allT;
            layer.InputNames = {'in1'}; %, 'in2'};
            layer.OutputNames = {'out1', 'out2', 'T'};

            % Set layer description.
            layer.Description = "apply constant transform to input";

            % Set layer type.
            layer.Type = "TFOffsetLayer";

            layer.T0 = zeros(4,4,length(allT));
            for i = 1:size(layer.T0, 3)
                layer.T0(:, :, i) = inv(allT{i});
            end

            layer.T0 = dlarray(layer.T0, 'SSCB');

            layer.h.structure.(layer.Name).indT = {};
        end

        function setIndT(layer, indT)
            layer.h.structure.(layer.Name).indT = indT;
            %             for i = 1:length(indT)
            %                 val = indT{i};
            %                 if isempty(val)
            %                     val = 0;
            %                 end
            %                 layer.h.structure.(layer.Name).indT(i) = val;
            %
            %             end
        end

        function T = getT(~, rpy, p)
            ct = cos(rpy);
            st = sin(rpy);
            R11 = ct(1, :).*ct(3, :).*ct(2, :) - st(1, :).*st(3, :);
            R12 = -ct(1, :).*ct(2, :).*st(3, :) - st(1, :).*ct(3, :);
            R13 = ct(1, :).*st(2, :);
            R21 = st(1, :).*ct(3, :).*ct(2, :) + ct(1, :).*st(3, :);
            R22 = -st(1, :).*ct(2, :).*st(3, :) + ct(1, :).*ct(3, :);
            R23 = st(1, :).*st(2, :);
            R31 = -st(2, :).*ct(3, :);
            R32 = st(2, :).*st(3, :);
            R33 = ct(2, :);

            %             T = [R11 R12 R13 p(1)
            %                 R21 R22 R23 p(2)
            %                 R31 R32 R33 p(3)
            %                 0   0   0   1];
            T = [R11; R21; R31; 0; R12; R22; R32; 0; R13; R23; R33; 0; p(1); p(2); p(3); 1];
        end

        function R = getR(~, rpy)
            rpy = reshape(rpy, 3, 1, []);
            ct = cos(rpy);
            st = sin(rpy);
            R11 = ct(1, :, :).*ct(3, :, :).*ct(2, :, :) - st(1, :, :).*st(3, :, :);
            R12 = -ct(1, :, :).*ct(2, :, :).*st(3, :, :) - st(1, :, :).*ct(3, :, :);
            R13 = ct(1, :, :).*st(2, :, :);
            R21 = st(1, :, :).*ct(3, :, :).*ct(2, :, :) + ct(1, :, :).*st(3, :, :);
            R22 = -st(1, :, :).*ct(2, :, :).*st(3, :, :) + ct(1, :, :).*ct(3, :, :);
            R23 = st(1, :, :).*st(2, :, :);
            R31 = -st(2, :, :).*ct(3, :, :);
            R32 = st(2, :, :).*st(3, :, :);
            R33 = ct(2, :, :);

            %             T = [R11 R12 R13 p(1)
            %                 R21 R22 R23 p(2)
            %                 R31 R32 R33 p(3)
            %                 0   0   0   1];
            R1 = cat(2, R11, R12, R13);
            R2 = cat(2, R21, R22, R23);
            R3 = cat(2, R31, R32, R33);
            R = cat(1, R1, R2, R3);
            %             R = [R11, R12, R13
            %                 R21, R22, R23
            %                 R31, R32, R33];
        end

        function [Z1, Z2, Tdlarray] = predict(layer, X)
            % X: xyz rpy (6 x batch) this is transform from world to camera
            % coordinates
            % Z: transform matrix (16 x batch)


            T = [];
            indT = layer.h.structure.(layer.Name).indT;
            if ~isempty(indT)
                X = X./10;
                %             X = dlarray(reshape(X, 6, []),'SB');
                batchSize = size(X, 3);
                curInd = layer.h.structure.curInd;
                indT = indT{curInd};
                %             assert(batchSize >= length(indT))
                %             indT(indT==0) = batchSize;


%                 for b = 1:length(indT)
                    if ~isempty(indT)
                        R = layer.getR(X(4:6, indT));
                        %             R = reshape(X(4:12, :), 3, 3, []) + eye(3);
                        R = pagemtimes(layer.T0(1:3, 1:3, indT), R);

                        %             R = pagemtimes(R2, R);

                        %             P = reshape(X(1:3, :, :), 3 ,1, []);
                        %             P = [1;1;1].*P;
                        %             P = pagemtimes(layer.T0(1:3, 1:3, :), P) + layer.T0(1:3,end,:);
                        %             P = reshape(X(1:3, :), 3 , 1, []) + layer.T0(1:3, end, :);
                        P = reshape(X(1:3, indT), 3 , 1, []) + layer.T0(1:3, end, indT);
                        T = [[R P]; repmat([0 0 0 1], 1, 1,  size(R, 3))];
                    else
                        T = eye(4);
                    end

%                     Tbatch = cat(3, Tbatch, T);
%                 end
            end
            %             T = [X(1) X(2) X(3) X(4)
            %                 X(5) X(6) X(7) X(8)
            %                 X(9) X(10) X(11) X(12)
            %                 0 0 0 1];
            %             T = T + layer.T0;
            Z = dlarray(T, 'SSB');
            Tdlarray = dlarray(T, 'SSB');

            Z1 = Z;% dlarray(Z, 'CB');
            Z2 = Z;% dlarray(Z, 'CB');
        end
        % %
        %         function [dLdX] = backward(layer, X, ~, dLdZ, ~)
        %
        %             dLdX = X*0;
        %
        %         end

    end
end
