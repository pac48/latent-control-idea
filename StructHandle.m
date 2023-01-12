classdef StructHandle < handle

    properties % Public Access
        structure = struct();
    end
    methods(Static)
        function lobj = loadobj(obj)
            lobj = obj;
        end
    end
    methods
        function sobj = saveobj(obj)
            sobj = obj;
        end
    end
end