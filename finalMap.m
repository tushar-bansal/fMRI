function output = finalMap(A,IdToFeatureMapCentered)

for i=1:size(A,1)
   
    for j = 1: size(IdToFeatureMapCentered,1);
        c(j) = norm(A(i,:)-IdToFeatureMapCentered(j,:));
    end
    [a b] = sort(c);
    output(i,:) = b;
end
