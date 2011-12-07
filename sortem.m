function [sortEigVecs,sortEigVals] = sortem(eigVecs,eigVals) 
 
diagonal = diag(eigVals); 
[diagonal, index] = sort(diagonal); 
index = flipud(index); 
  
pcaEigVals = zeros(size(eigVals)); 
for i = 1 : size(eigVals, 1) 
    sortEigVals(i, i) = eigVals(index(i), index(i)); 
    sortEigVecs(:, i) = eigVecs(:, index(i)); 
end;