function classification = classif(Ytrain, Ytest) 
% classification = classify(Ytrain, Ytest) 
% 
% Given the train matrix Ytrain and the test matrix Ytest, 
% this function returs a vector classification, where  
% for Ytest(:, a), the nearest element of Ytrain is 
% Ytrain(:, classification(a)). 
 
distances = dist(Ytrain', Ytest); 
classification = zeros(size(Ytest,2),1); 
for a=1:size(Ytest,2), 
   aux = find(distances(:,a)==min(distances(:,a))); 
   classification(a) = aux(1); 
end 

