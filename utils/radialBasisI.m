function mVal = radialBasisI(q, q0, c, width)
%             https://en.wikipedia.org/wiki/Error_function
mVal = -sqrt(width*pi)*.5*(erfc(sqrt(1/width)*(q - c) ) - erfc(sqrt(1/width)*(q0 - c)));
end