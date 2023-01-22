function gradients = thresholdL2Norm(gradients, totalGradientThreshold, objects)
for ind = 1:length(objects)
    object = objects{ind};
    totalGrad = 0;
    gradInds = [];
    for i = 1:size(gradients,1)
        row = gradients(i, :);
        if contains(row.Layer, object)
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
    %
    % gradientNorm = sqrt(sum(gradients(:).^2));
    %     if gradientNorm > gradientThreshold
    %         gradients = gradients * (gradientThreshold / gradientNorm);
    %     end
    %
    %     gradients.Value = gradients.Value
end
end