clc
clear

% flags

num_e_vecs_to_use = 48;
train_set_size = 1000;
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


% Sizes

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
    fprintf('Normalizing Images...%d\n',cputime-t)
    
    % get hist of first image
    ref_hist = imhist(reshape(data_train(1,:,:),imgRows,imgCols));

    % rewrite images histogram equalized to reference histogram
    for i=1:size(data_train,1)
        data_train(i,:,:) = histeq(reshape(data_train(i,:,:),imgRows,imgCols),ref_hist); 
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

% TODO: combine the two loops below

% Compute modified faces
vec = vec(:,1:num_e_vecs_to_use);
features_train = zeros(trainInstances, imgRows, num_e_vecs_to_use);
for i=1:trainInstances
	features_train(i,:,:) = reshape(faces_train(i,:,:), imgRows, imgCols)*vec;
end

% Resize 
features_train_reshaped = zeros(trainInstances, nVars);
for i=1:trainInstances
	features_train_reshaped(i,:) = reshape(features_train(i,:,:), nVars, 1);
end
features_train = features_train_reshaped;


%
% Multi-class Linear SVM
%

fprintf('Compute SVM... %d\n',cputime-t)

lambda = 1e-2;

funObj = @(w)SSVMMultiLoss(w,features_train,labels_train,nClasses);
wLinear = minFunc(@penalizedL2,zeros(nVars*nClasses,1),options,funObj,lambda);
wLinear = reshape(wLinear,[nVars nClasses]);

%
% Compute Error
%

fprintf('Compute Error...%d\n',cputime-t)

% Training
[junk yhat] = max(features_train*wLinear,[],2);
trainErr_linear = sum(labels_train~=yhat)/length(labels_train);

% Test

[y,junk] = find(labels_test');

% TODO: combine both loops

features_test = zeros(testInstances, size(faces_test, 2), num_e_vecs_to_use);
for i=1:testInstances
	features_test(i,:,:) = reshape(faces_test(i,:,:), imgRows, imgCols)*vec;
end

features_train_reshaped = zeros(testInstances, nVars);
for i=1:testInstances
	features_train_reshaped(i,:) = reshape(features_test(i,:,:), nVars, 1);
end
features_test = features_train_reshaped;

[junk yhat_test] = max(features_test*wLinear,[],2);
testErr = sum(y~=yhat_test)/length(y);


% Useful Functions
%{

% show specific image
imshow(reshape(data_train(1,:,:),48,48));

%}