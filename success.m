function rate=success(classification, stest, strain) 
% success(classification, stest, strain) 
% 
% Returns the success rate of the classification. 
%  
% 'classification': vector created by function 'classify()', 
% 'imgpperson': number of images per person in the image database, 
% 'rate': success rate (0 <= rate <=1). 
 
error = 0; 
succe = 0; 
for c=1: length(classification), 
   if (floor(floor((c-1)/stest)*strain) < classification(c)) & (classification(c) <= floor((1+floor((c-1)/stest))*strain)) 
      succe = succe+1; 
   else 
      error = error+1; 
   end 
end 
 
rate = succe/length(classification);