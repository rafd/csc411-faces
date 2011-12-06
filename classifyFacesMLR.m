function[targets_test] = classifyFacesMLR(wSoftmax,data_test,nClasses)
% Evaluates an MLR classifier for the Toronto Faces Dataset on test data
% Reshape parameters from a vector to a matrix, the last column should
% contain zeros in this version (don't worry about why, it's basically
% removing redundancy in the original softmax formulation)
[num_test,num_rows,num_cols] = size(data_test);
nVars = num_rows*num_cols;
wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
wSoftmax = [wSoftmax zeros(nVars+1,1)];

% Make predictions on the test set
X_test = reshape(data_test,num_test,num_rows*num_cols);
X_test = [ones(num_test,1), X_test];

[junk targets_test] = max(X_test*wSoftmax,[],2);