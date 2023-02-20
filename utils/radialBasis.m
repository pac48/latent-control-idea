function mVal = radialBasis(q, c, width)
mVal = exp(-1/width*(q - c).^2);
end
