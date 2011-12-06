function[wSoftmax] = trainFacesMLRClassifier(data_train,targets_train,data_unlabeled)
% Trains a multinomial logistic regression classifier to predict the labels
% for data_test. Note that this is purely supervised and does not use
% unlabeled data in any way.
[num_train,num_rows,num_cols] = size(data_train);

% Reshape the data to the standard num_data x num_dimensions format
X = reshape(data_train,num_train,num_rows*num_cols);

% Add bias
X = [ones(num_train,1), X];

% Convert targets from 1 of K to vector representation
[junk,y] = max(targets_train,[],2);
nClasses = size(targets_train,2);
nVars = num_rows*num_cols;

% Create a pointer to the classifier function
funObj = @(W)SoftmaxLoss2(W,X,y,nClasses);
fprintf('Training multinomial logistic regression model...\n');
options = [];

% This version of softmax regression only requires (nVars+1)x(nClasses-1)
% parameters as opposed to (nVars+1)x(nClasses)
wInit = zeros((nVars+1)*(nClasses-1),1);
wSoftmax = minFunc(funObj,wInit,options);