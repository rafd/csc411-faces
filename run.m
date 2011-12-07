clc
clear

% flags

num_e_vecs_to_use = 48;
max_val_of_train_set_on_which_to_train = 2000;
flag_load_normalized_data = 1;
flag_normalize_images = 0;
options.Display = 'none';


t = cputime;

% Load TFD Dataset
% provides:
%   data_train: 2925x48x48 grayscale images
%   targets_train: 2925x7 labels

fprintf('Loading Data...\n');
if flag_load_normalized_data 
    load TFD_hw4_train_norm
else
    load TFD_hw4_train
end

nClasses = size(targets_train,2); % 7
facedatabase = double(data_train(1:max_val_of_train_set_on_which_to_train,:,:));
targets = targets_train(1:max_val_of_train_set_on_which_to_train,:);

% Sort

[y,x] = find(targets');

sort_index = sortrows([x,y],2);

sorted_labels = sort_index(:,2);

sorted_faces = zeros(size(facedatabase));

for i=1:size(facedatabase,1)

    sorted_faces(i,:,:) = facedatabase(sort_index(i,1),:,:);

end

facedatabase = sorted_faces;
targets = sorted_labels;

%targets = y;

% 1: Normalize Histogram of Images

if flag_normalize_images && ~flag_load_normalized_data
    fprintf('Normalizing Images...%d\n',cputime-t)
    
    % get hist of first image
    ref_hist = imhist(reshape(data_train(1,:,:),image_rows,image_cols));

    % rewrite images histogram equalized to reference histogram
    for i=1:size(data_train,1)
        data_train(i,:,:) = histeq(reshape(data_train(i,:,:),image_rows,image_cols),ref_hist); 
    end
else
    % do nothing
end


% Ref Variables

[nInstances, image_rows, image_cols] = size(facedatabase); 


% 2: 2D-LDA

fprintf('Computing Features...\n');

% Massage the data into the form desired by the 2D LDA function
nclass = 7; 
height = 48; 
width = 48; 
 
for i=1:size(facedatabase, 1)
    trainSample{i}=reshape(facedatabase(i,:,:), height, width); 
end 
 
for i=1:size(facedatabase, 1)
    testSample{i}=reshape(facedatabase(i,:,:), height, width); 
end

[vec, val] = tdfda(trainSample, nclass, targets);

% 3: Compute modified faces
vec = vec(:,1:num_e_vecs_to_use);
modified_faces = zeros(size(facedatabase, 1), size(facedatabase, 2), num_e_vecs_to_use);
for i=1:size(facedatabase, 1)
	%modified_faces(i,:,:) = reshape(facedatabase(i,:,:), height, width)*vec;
	modified_faces(i,:,:) = reshape(facedatabase(i,:,:), height, width);
end

nVars = height * num_e_vecs_to_use;
new_modified_faces = zeros(size(modified_faces, 1), nVars);
for i=1:size(modified_faces, 1)
	new_modified_faces(i,:) = reshape(modified_faces(i,:,:), nVars, 1);
end
modified_faces = new_modified_faces;
% 4: Multi-class Linear SVM

fprintf('Compute SVM... %d\n',cputime-t)

%X = double(data_train(:,1:nVars));

% return the class # of the targets
lambda = 1e-2;

% Linear
funObj = @(w)SSVMMultiLoss(w,modified_faces,targets,nClasses);
wLinear = minFunc(@penalizedL2,zeros(nVars*nClasses,1),options,funObj,lambda);

fprintf('Reshape... %d\n',cputime-t)
wLinear = reshape(wLinear,[nVars nClasses]);

% compute error
fprintf('Compute Error...%d\n',cputime-t)
[junk yhat] = max(modified_faces*wLinear,[],2);
trainErr_linear = sum(targets~=yhat)/length(targets);

% Compute test error
[y,junk] = find(targets_train(max_val_of_train_set_on_which_to_train+1:2925,:,:)');
facedatabase_test = double(data_train(max_val_of_train_set_on_which_to_train+1:2925,:,:));
modified_faces_test = zeros(size(facedatabase_test, 1), size(facedatabase_test, 2), num_e_vecs_to_use);

for i=1:size(facedatabase_test, 1)
	modified_faces_test(i,:,:) = reshape(facedatabase_test(i,:,:), height, width)*vec;
end
new_modified_faces = zeros(size(modified_faces_test, 1), nVars);
for i=1:size(modified_faces_test, 1)
	new_modified_faces(i,:) = reshape(modified_faces_test(i,:,:), nVars, 1);
end
modified_faces_test = new_modified_faces;
[junk yhat_test] = max(modified_faces_test*wLinear,[],2);
testErr = sum(y~=yhat_test)/length(y);


% Useful Functions
%{

% show specific image
imshow(reshape(data_train(1,:,:),48,48));

%}