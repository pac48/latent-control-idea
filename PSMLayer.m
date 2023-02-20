classdef PSMLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        numBasis
        numSamples
        numDims

        axis
        A
        Ad
    end
    properties(Learnable)
        weights
        T_object_goal
    end
    methods
        function layer = PSMLayer(skillName, numDims, numBasis, numSamples)
            % Create a PSMLayer.

            % Set layer name.
            layer.Name = ['PSMLayer_' skillName];

            layer.numBasis = numBasis;
            layer.numSamples = numSamples;
            layer.numDims = numDims;

            % Set layer description.
            layer.Description = "generate PSM trajectory from predicted input parameters";

            layer.OutputNames = {'psm', 'psmD', 'axis','Trobot_goal', 'Trobot_object'};

            % Set layer type.
            layer.Type = "PSMLayer";

            layer.weights = dlarray(rand(numBasis, 1, numDims)-.5, 'SSB');
            layer.T_object_goal = dlarray(zeros(6,1), 'CB');

            axisLenth = 2;
            layer.axis = linspace(-axisLenth, 0, layer.numSamples);
            width = axisLenth/200;
            c = linspace(layer.axis(1)-.5, layer.axis(end)+.5, layer.numBasis-1);

            A = ones(length(layer.axis), layer.numBasis);
            A(:, 1:layer.numBasis-1) = radialBasis(layer.axis', c, width);
            
            Ad = zeros(length(layer.axis), layer.numBasis);
            Ad(:, 1:layer.numBasis-1) = radialBasisD(layer.axis', c, width);

            layer.A = dlarray(A, 'SSB');
            layer.Ad = dlarray(Ad, 'SSB');            
            layer.axis = dlarray(layer.axis, 'SB');

        end

        function [psm, psmD, axisOut, Trobot_goal, Trobot_objectOut] = predict(layer, Trobot_object)
            % Trobot_object is predicted by nerf
            psm = dlarray([], 'SSB');
            psmD = dlarray([], 'SSB');
            Trobot_goal = dlarray([], 'SSB');
            Trobot_objectOut = dlarray([], 'SSB');
            axisOut = dlarray([], 'SB');

            if isempty(Trobot_object)
                return
            end
            Trobot_object = extractdata(Trobot_object);
            Trobot_object = dlarray(Trobot_object, 'SSB');

            if size(Trobot_object, 1) == 6 % convert to T
                Trobot_object = reshape(Trobot_object, 6,1,[]);
                Trobot_object = getT(Trobot_object(1:3,:), Trobot_object(4:6,:));
            else
                Trobot_object = reshape(Trobot_object, 4, 4, []);
            end

            T_object_goal_  = getT(layer.T_object_goal(1:3,:), layer.T_object_goal(4:6,:));
            Trobot_goal = pagemtimes(Trobot_object, T_object_goal_);

          
            %TODO: need to add radial basis using learnable weights
%             psm = repmat(layer.axis + Trobot_object(1,1,:), layer.numDims, 1);
            
            x = pagemtimes(stripdims(layer.A), layer.weights);
            xd = pagemtimes(stripdims(layer.Ad), layer.weights);
            

            psm = dlarray(reshape(x, size(layer.A,1), []), 'SSB');
            psmD = dlarray(reshape(x, size(layer.Ad,1), []), 'SSB');
            axisOut = dlarray(layer.axis, 'SB');
            Trobot_goal = dlarray(Trobot_goal, 'SSB');
            Trobot_objectOut = dlarray(Trobot_object, 'SSB');

        end

    end
end
