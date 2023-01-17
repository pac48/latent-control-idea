classdef TFAllignLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties(Learnable)
        Trobot_background;
        Tobj_grasp;
        scale1;
        scale2;
    end

    methods(Static)

        function Tinv = TFInv(T)
            T = stripdims(T);
            R = T(1:3, 1:3, :);
            R = permute(R, [2 1 3]);
            P = -pagemtimes(R, T(1:3, end, :));
            ONE = ones(1, 1, size(R, 3));
            ZERO = zeros(1, 1, size(R, 3));
            Tinv = [[R P]; ZERO ZERO ZERO ONE];

        end
    end

    methods
        function layer = TFAllignLayer(name)
            % Create a TFAllignLayer

            % Set layer name.
            layer.Name = name;
            layer.InputNames = {'in1', 'in2'};
            layer.OutputNames = {'Trobot_grasp', 'Trobot_cam', 'Tcam_grasp', 'Tgrasp_robot'}; %{'T1robot_cam', 'T2robot_cam', 'Trobot_obj', 'Tgrasp_cam'};

            % Set layer description.
            layer.Description = "Allign 3D transforms to each other";

            % Set layer type.
            layer.Type = "TFAllignLayer";

            layer.Trobot_background = zeros(6, 1);
            layer.Tobj_grasp = zeros(6, 1);
            layer.scale1 = 1;
            layer.scale2 = 1;

        end


        function [Trobot_grasp, Trobot_cam, Tcam_grasp, Tgrasp_robot] = predict(layer, Tbackground_cam, Tcam_obj)
            Trobot_grasp = dlarray([], 'SSB');

            if isempty(Tbackground_cam) || isempty(Tcam_obj)
                return
            end

            if size(Tbackground_cam, 1) == 6 % convert to T
                Tbackground_cam = reshape(Tbackground_cam, 6,1,[]);
                Tbackground_cam = getT(Tbackground_cam(1:3, 1 ,:), Tbackground_cam(4:6, 1 ,:));
            else
                Tbackground_cam = reshape(Tbackground_cam, 4, 4, []);
            end
            Tcam_background = TFAllignLayer.TFInv(Tbackground_cam);


            if size(Tcam_obj, 1) == 6 % convert to T
                Tcam_obj = reshape(Tcam_obj, 6,1,[]);
                Tcam_obj = getT(Tcam_obj(1:3, 1 ,:), Tcam_obj(4:6, 1 ,:));
            else
                Tcam_obj = reshape(Tcam_obj, 4, 4, []);
            end
            Tobj_cam = TFAllignLayer.TFInv(Tcam_obj);


            %             batchSize = size(Tobj_cam, 3);
            %             assert(batchSize == size(Trobot_grasp, 3))

            Trobot_background = layer.Trobot_background;
            Trobot_background = reshape(Trobot_background, 6, 1 ,[]);
            Trobot_background = getT(Trobot_background(1:3, 1 ,:), Trobot_background(4:6, 1 ,:));
         
            Tobj_grasp = layer.Tobj_grasp;
            Tobj_grasp = reshape(Tobj_grasp, 6,1,[]);
            Tobj_grasp = getT(Tobj_grasp(1:3, 1 ,:), Tobj_grasp(4:6, 1 ,:));
         

%             S1 = [[ones(3) repmat(layer.scale1, 3,1)]; [0 0 0 1]];
%             S2 = [[ones(3) repmat(layer.scale2, 3,1)]; [0 0 0 1]];

%             part1 = pagemtimes(Trobot_background, ((S2).*Tbackground_cam));
%             part2 = pagemtimes(((S1).*Tcam_obj), Tobj_grasp);


%              S1 = [1 1 1 exp(layer.scale1)
%                   1 1 1 exp(layer.scale1)
%                   1 1 1 exp(layer.scale1)
%                   1 1 1 1];
%              S2 = [1 1 1 exp(layer.scale2)
%                   1 1 1 exp(layer.scale2)
%                   1 1 1 exp(layer.scale2)
%                   1 1 1 1];
%             part1 = pagemtimes(Trobot_background, Tbackground_cam.*S1);
%             part2 = pagemtimes(Tcam_obj.*S2, Tobj_grasp);

%             

            % this converges but the results don't make sense
%             layer.scale1 = .1*layer.scale1+.9;
%             layer.scale2 = .1*layer.scale2+.9;
            layer.scale1 = layer.scale1*0.001 + 1/1.6467;
            layer.scale2 = layer.scale2*0.001 + 1/0.4958;

            S1 = [layer.scale1 0 0 0
                  0 layer.scale1 0 0 
                  0 0 layer.scale1 0 
                  0 0 0 1/layer.scale1];

             S2 = [layer.scale2 0 0 0
                  0 layer.scale2 0 0 
                  0 0 layer.scale2 0
                  0 0 0 1/layer.scale2];

             S1Inv = [1/layer.scale1 0 0 0
                  0 1/layer.scale1 0 0 
                  0 0 1/layer.scale1 0 
                  0 0 0 layer.scale1];

             S2Inv = [1/layer.scale2 0 0 0
                  0 1/layer.scale2 0 0 
                  0 0 1/layer.scale2 0
                  0 0 0 layer.scale2];


%              S1 = [1 1 1 layer.scale1
%                   1 1 1 layer.scale1
%                   1 1 1 layer.scale1
%                   1 1 1 1];
%              S2 = [1 1 1 layer.scale2
%                   1 1 1 layer.scale2
%                   1 1 1 layer.scale2
%                   1 1 1 1];

%             Trobot_cam = pagemtimes(Trobot_background, Tbackground_cam.*S1);
%             Tcam_grasp = pagemtimes(Tcam_obj.*S2, Tobj_grasp);
%             Trobot_cam = pagemtimes(pagemtimes(Trobot_background, S1), pagemtimes(Tbackground_cam, S2));
%             Tcam_grasp = pagemtimes(Tcam_obj, Tobj_grasp);

            Trobot_cam = pagemtimes(Trobot_background*S1, Tbackground_cam);
            Tcam_grasp = pagemtimes(Tcam_obj, Tobj_grasp*S2);
            Trobot_grasp =  pagemtimes(Trobot_cam, Tcam_grasp);

            Tbackground_robot = TFAllignLayer.TFInv(Trobot_background);
            Tgrasp_obj = TFAllignLayer.TFInv(Tobj_grasp);
            Tcam_robot = pagemtimes(Tcam_background, S1Inv*Tbackground_robot);
            Tgrasp_cam = pagemtimes(S2Inv*Tgrasp_obj, Tobj_cam);
            Tgrasp_robot = pagemtimes(Tgrasp_cam, Tcam_robot);


            Trobot_grasp = dlarray(Trobot_grasp, 'SSB');
            Trobot_cam = dlarray(Trobot_cam, 'SSB');
            Tcam_grasp = dlarray(Tcam_grasp, 'SSB');
            Tgrasp_robot = dlarray(Tgrasp_robot, 'SSB');


             

%             Trobot_grasp =  (Trobot_background)*((scale)Tbackground_cam)*((scale)*Tcam_obj)*(Tobj_grasp)


%             Tgrasp_robot = (Tgrasp_obj)*((scale)*Tobj_cam)*((scale)Tcam_background)*(Tbackground_robot)



        end

    end
end
