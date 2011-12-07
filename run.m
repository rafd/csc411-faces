clc
clear

% flags

num_e_vecs_to_use = 10;
train_set_size = 2500;
flag_load_normalized_data = 1;
flag_normalize_images = 0;
options.Display = 'none';


t = cputime;


%
% Load TFD Dataset
%

% provides:
%   data_train: 2925x48x48 grayscale images
%   targets_train: 2925x7 labels

fprintf('Loading Data...\n');
if flag_load_normalized_data 
    load TFD_hw4_train_norm
else
    load TFD_hw4_train
end


%
% Subset into Training and Test Sets
%

faces_train = double(data_train(1:train_set_size,:,:));
faces_test = double(data_train(train_set_size+1:size(data_train,1),:,:));
labels_train = targets_train(1:train_set_size,:);
labels_test = targets_train(train_set_size+1:size(targets_train,1),:);

%
% Define Some Useful Sizes
%

nClasses = size(labels_train,2); % 7
[trainInstances, imgRows, imgCols] = size(faces_train);
testInstances = size(faces_test,1);
nVars = imgRows * num_e_vecs_to_use;

%
% Sort Training Data
%

[y,x] = find(labels_train');
sorted_indices = sortrows([x,y],2);
sorted_labels = sorted_indices(:,2);

sorted_faces = zeros(size(faces_train));
for i=1:trainInstances
    sorted_faces(i,:,:) = faces_train(sorted_indices(i,1),:,:);
end

faces_train = sorted_faces;
labels_train = sorted_labels;


%
% Normalize Histogram of Images
%

if flag_normalize_images && ~flag_load_normalized_data
    fprintf('Normalizing Images...\n')
    
    % get hist of first image
    ref_hist = imhist(reshape(data_train(1,:,:),imgRows,imgCols));

    % rewrite images histogram equalized to reference histogram
    for i=1:size(data_train,1)
        data_train(i,:,:) = medfilt2(histeq(reshape(data_train(i,:,:),imgRows,imgCols),ref_hist)); 
    end
end

%
% Compute Features with 2D-LDA
%

fprintf('Computing Features...\n');

% Resize into the form desired by the 2D LDA function
for i=1:trainInstances
    trainSample{i}=reshape(faces_train(i,:,:), imgRows, imgCols); 
end

[vec, val] = tdfda(trainSample, nClasses, labels_train);

% Compute modified faces
vec = vec(:,1:num_e_vecs_to_use);

features_train = zeros(trainInstances, nVars);
for i=1:trainInstances
	features_train(i,:,:) = reshape((reshape(faces_train(i,:,:), imgRows, imgCols)*vec),nVars,1);
end


%
% Multi-class Linear SVM
%
%{
fprintf('Compute SVM...\n')

lambda = 1e-2;

funObj = @(w)SSVMMultiLoss(w,features_train,labels_train,nClasses);
wLinear = minFunc(@penalizedL2,zeros(nVars*nClasses,1),options,funObj,lambda);
wLinear = reshape(wLinear,[nVars nClasses]);
%}

%
% Multinomial Logistic Regression
%

X = [ones(trainInstances,1) features_train];
funObj = @(W)SoftmaxLoss2(W,X,labels_train,nClasses);

lambda = 1e-4*ones(nVars+1,nClasses-1);
lambda(1,:) = 0; % Don't penalize biases
fprintf('Training multinomial logistic regression model...\n');
wSoftmax = minFunc(@penalizedL2,zeros((nVars+1)*(nClasses-1),1),options,funObj,lambda(:));
wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
wSoftmax = [wSoftmax zeros(nVars+1,1)];

%
% Compute Error
%

fprintf('Compute Error...\n')

% Training
[junk yhat] = max(X*wSoftmax,[],2);
trainErr = sum(yhat~=labels_train)/length(labels_train);

% Test

% Test

[y,junk] = find(labels_test');
features_test = zeros(testInstances, nVars);
for i=1:testInstances
	features_test(i,:,:) = reshape((reshape(faces_test(i,:,:), imgRows, imgCols)*vec), nVars, 1);
end

[junk yhat_test] = max([ones(testInstances,1) features_test]*wSoftmax,[],2);
testErr = sum(yhat_test~=y)/length(y);

fprintf('Run Time:       %3.0f\n',cputime-t)
fprintf('Train Error: %6.2f\n', trainErr)
fprintf('Test Error:  %6.2f\n', testErr)

%
% Confusion Matrix
%

% Train

confusion_matrix_train = zeros(nClasses,nClasses);
for i=1:trainInstances
    confusion_matrix_train(labels_train(i),yhat(i)) = confusion_matrix_train(labels_train(i),yhat(i)) + 1;
end

% Test
confusion_matrix_test = zeros(nClasses,nClasses);
for i=1:testInstances
    confusion_matrix_test(y(i),yhat_test(i)) = confusion_matrix_test(y(i),yhat_test(i)) + 1;
end

% Useful Functions
%{

% show specific image
imshow(reshape(data_train(1,:,:),48,48));

%}