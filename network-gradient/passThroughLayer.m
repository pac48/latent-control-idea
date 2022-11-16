classdef passThroughLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.
        w
    end

    properties (State)
        % (Optional) Layer state parameters.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state parameters.
    end

    methods
        function layer = passThroughLayer()
            layer.Name = 'pass_through_layer';
        end

        function layer = initialize(layer,layout)
            % (Optional) Initialize layer learnable and state parameters.
            
        end


        function Z = predict(layer,X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN,
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state
            %    with state1,...,stateK, where K is the number of state
            %    parameters.
            Z=X;

        end

        function [Z,state] = forward(layer,X)
            % (Optional) Forward input data through the layer at training
            % time and output the result, the updated state, and a memory
            % value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Layer input data
            % Outputs:
            %         Z      - Output of layer forward function
            %         state  - (Optional) Updated layer state
            %         memory - (Optional) Memory value for custom backward
            %                  function
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN,
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state
            %    with state1,...,stateK, where K is the number of state
            %    parameters.
            Z=X;
            state = [];
            %             memory = [];
        end



        function [dLdZ, dLdW] = backward(layer,X,Z,dLdZ,dLdSout)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer   - Layer to backward propagate through
            %         X       - Layer input data
            %         Z       - Layer output data
            %         dLdZ    - Derivative of loss with respect to layer
            %                   output
            %         dLdSout - (Optional) Derivative of loss with respect
            %                   to state output
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX   - Derivative of loss with respect to layer input
            %         dLdW   - (Optional) Derivative of loss with respect to
            %                  learnable parameter
            %         dLdSin - (Optional) Derivative of loss with respect to
            %                  state input
            %
            %  - For layers with state parameters, the backward syntax must
            %    include both dLdSout and dLdSin, or neither.
            %  - For layers with multiple inputs, replace X and dLdX with
            %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
            %    the number of inputs.
            %  - For layers with multiple outputs, replace Z and dlZ with
            %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
            %    number of outputs.
            %  - For layers with multiple learnable parameters, replace
            %    dLdW with dLdW1,...,dLdWP, where P is the number of
            %    learnable parameters.
            %  - For layers with multiple state parameters, replace dLdSin
            %    and dLdSout with dLdSin1,...,dLdSinK and
            %    dLdSout1,...,dldSoutK, respectively, where K is the number
            %    of state parameters.
            dLdW = [];
            global calcJac
            if (calcJac)
                global jac
                jac = dLdZ;
            end
            %             dLdX = dLdZ;
        end
    end
end