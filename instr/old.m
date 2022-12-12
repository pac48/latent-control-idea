classdef Old < handle
    % Dense Pose  wrapper

    properties
        model
        kd
        numVerts
        allVertInds
        efficientNet
    end
    properties(Access=private)
        X
        Y
    end

    methods
        function obj = Old()
            [status,cmdout] = system('conda info -e');
            str = cmdout;
            expression = 'instr38 + ((\/.+)+)';
            [tokens,matches] = regexp(str,expression,'tokens','match');
            path = strip(tokens{1}{1});

            setenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
            terminate(pyenv)
            pyenv('Version', fullfile('/usr', 'bin', 'python3.10'), "ExecutionMode","OutOfProcess")
%              pyenv('Version', fullfile(path, 'bin', 'python'), "ExecutionMode","InProcess")
%             py.importlib.import_module('pytorch_lightning')
            mod = py.importlib.import_module('instr_matlab');
%             mod = py.importlib.import_module('matlab_test');
            py.importlib.reload(mod);

            chkpt = fullfile(pwd, 'pretrained_instr', 'models', 'pretrained_model.pth');
            mod.load_model(chkpt)

        end

      

        function [vertices, cameraTranslation] = predict(obj, img)
%            imgPy = py.numpy.array(img);
            [bboxes, ~, ~] = obj.efficientNet.predict(img,render=false);
            bboxes = double(floor(bboxes));
            vertices = []; 
            cameraTranslation = []; 
            if isempty(bboxes)
                return
            end
%             imshow(img(bboxes(2):bboxes(2)+bboxes(4), bboxes(1):bboxes(1)+bboxes(3),:))
            y0 = max(bboxes(1,2), 1);
            y1 = min(bboxes(1,2)+bboxes(1,4), size(img,1));
            x0 = max(bboxes(1,1), 1);
            x1 = min(bboxes(1,1)+bboxes(1,3), size(img,2));


            img = img(y0:y1, x0:x1, :);
            imgPy = py.numpy.array(img);
            outPy = cell(py.DecoMR.deco_mr.execute(imgPy));
            vertices = double(outPy{1});
            vertices = vertices(:, [3 1 2]);
            vertices(:, [1 3]) = -vertices(:, [1 3]);
            cameraTranslation = double(outPy{2});

        end

      
    end
end