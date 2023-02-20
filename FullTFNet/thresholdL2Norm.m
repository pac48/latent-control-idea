function gradients = thresholdL2Norm(gradients, totalGradientThreshold, objects)
for ind = 1:length(objects)
    object = objects{ind};
    totalGrad = 0;
    gradInds = [];
    for i = 1:size(gradients,1)
        row = gradients(i, :);
        if contains(row.Layer, object) && ~contains(row.Layer, 'PSM')
            gradInds = cat(1, gradInds, i);
            if strcmp(row.Parameter, "Bias")
                val = row.Value;
                val = extractdata(val{1});
                totalGrad = totalGrad + norm(val);
            end
        end
    end
    scale = 1;
    if totalGrad > totalGradientThreshold
        scale = totalGradientThreshold/totalGrad;
    end



    values = gradients.Value(gradInds);
    for i = 1:length(values)
        value = values{i};
        value = scale*value;
        values{i} = value;
    end
    gradients.Value(gradInds) = values;
 
end


for ind = 1:length(objects)
    object = objects{ind};
    totalGrad = 0;
    gradInds = [];
    for i = 1:size(gradients,1)
        row = gradients(i, :);
        if contains(row.Layer, object) && contains(row.Layer, 'PSM')
            gradInds = cat(1, gradInds, i);
            if strcmp(row.Parameter, "weights")
                val = row.Value;
                val = extractdata(val{1});
                totalGrad = totalGrad + sqrt(sum(val.^2, 'all'));
            end
        end
    end
    scale = 1;
    totalGradientThreshold2 = .1;
    if totalGrad > totalGradientThreshold2
        scale = totalGradientThreshold2/totalGrad;
    end

    values = gradients.Value(gradInds);
    for i = 1:length(values)
        value = values{i};
        value = scale*value;
        values{i} = value;
    end
    gradients.Value(gradInds) = values;
 
end


end