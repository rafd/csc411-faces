% This script will train a softmax regression classifier on the Toronto
% Faces Dataset.

model = 'SVM';

% Load the training set
load TFD_hw4_train;
% We don't use any unlabeled data here
data_unlabeled = 0;
% Convert the data to double otherwise softmax loss throws an error
data_train = double(data_train);

[num_faces,num_rows,num_cols] = size(data_train);

% Randomly split the data into a training a validation set
rand('state',1);
randn('state',1);
ind = randperm(num_faces);
num_train = ceil(num_faces*(4/5));
num_valid = num_faces - num_train;
data_valid = data_train(ind(num_train+1:num_faces),:,:);
targets_valid = targets_train(ind(num_train+1:num_faces),:);
data_train = data_train(ind(1:num_train),:,:);
targets_train = targets_train(ind(1:num_train),:);

% Learn the parameters on the training set
switch model
	case '2DLDA'
		train2DLDA(data_train, targets_train);
	case 'MLR'
		wSoftmax = trainFacesMLRClassifier(data_train,targets_train,data_unlabeled);
	case 'SVM'
		wSoftmax = trainFacesSVMClassifier(data_train,targets_train,data_unlabeled);
end

% Save the learned model to a mat file
save trainedModel wSoftmax;
% Evaluate the learned parameters on the validation set and report the
% validation set accuracy
switch model
	case MLR
		[targets_test] = classifyFacesMLR(wSoftmax,data_valid,size(targets_valid,2));
	case SVM
		[targets_test] = classifyFacesMLR(wSoftmax,data_valid,size(targets_valid,2));
end
targets_valid = oneOfK2Num(targets_valid);
validAccuracy = sum(sum(targets_test == targets_valid))/size(targets_valid,1)

% Save the results to a mat file
save test.mat targets_test