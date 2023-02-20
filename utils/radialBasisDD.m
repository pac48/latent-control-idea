function mVal = radialBasisDD(q, c, width)
mVal = (4/width^2*(q - c).^2 - 2/width).*exp(-1/width*(q - c).^2);
end