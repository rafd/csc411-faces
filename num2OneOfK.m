function[oneOfKLabels] = num2OneOfK(labels)
% Converts a vector of labels in the form 1,2,...,K to a 1 of K
% representation
oneOfKLabels = cumsum(ones(size(labels,1),max(labels)),2);
oneOfKLabels = oneOfKLabels==repmat(labels,1,max(labels));