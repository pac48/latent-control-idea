function mVal = radialBasisD(q, c, width)
mVal = -(2/width*(q - c)).*exp(-1/width*(q - c).^2);
end