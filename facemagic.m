clc
clear

% flags

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

% 1: Normalize Histogram of Images

if flag_normalize_images && ~flag_load_normalized_data
    fprintf('Normalizing Images...%d\n',cputime-t)
    
    % get hist of first image
    ref_hist = imhist(reshape(data_train(1,:,:),image_rows,image_cols));

    % rewrite images histogram equalized to reference histogram
    for i=1:image_count
        data_train(i,:,:) = histeq(reshape(data_train(i,:,:),image_rows,image_cols),ref_hist); 
    end
else
    % do nothing
end


% Ref Variables

[nInstances, image_rows, image_cols] = size(data_train);
nVars = 5; 
nClasses = size(targets_train,2); % 7


% 2: 2D-LDA

fprintf('Computing Features...\n');





% 3: Multi-class Linear SVM

fprintf('Compute SVM... %d\n',cputime-t)

X = double(data_train(:,1:nVars));

% return the class # of the targets
[y,junk] = find(targets_train');
lambda = 1e-2;

% Linear
funObj = @(w)SSVMMultiLoss(w,X,y,nClasses);
wLinear = minFunc(@penalizedL2,zeros(nVars*nClasses,1),options,funObj,lambda);

fprintf('Reshape... %d\n',cputime-t)
wLinear = reshape(wLinear,[nVars nClasses]);

% compute error
fprintf('Compute Error...%d\n',cputime-t)
[junk yhat] = max(X*wLinear,[],2);
trainErr_linear = sum(y~=yhat)/length(y);



% Useful Functions
%{

% show specific image
imshow(reshape(data_train(1,:,:),48,48));

%}
