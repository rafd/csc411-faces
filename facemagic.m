% flags

flag_load_normalized_data = 1;
flag_normalize_images = 0;


% Load TFD Dataset
% provides:
%   data_train: 2925x48x48 grayscale images
%   targets_train: 2925x7 labels

if flag_load_normalized_data 
    load TFD_hw4_train_norm
else
    load TFD_hw4_train
end

% 1: Normalize Histogram of Images

if flag_normalize_images && ~flag_load_normalized_data
    % get hist of first image
    ref_hist = imhist(reshape(data_train(1,:,:),image_rows,image_cols));

    % rewrite images histogram equalized to reference histogram
    for i=1:image_count
        data_train(i,:,:) = histeq(reshape(data_train(i,:,:),image_rows,image_cols),ref_hist); 
    end
else
    % do nothing
end


[image_count, image_rows, image_cols] = size(data_train);


% 2: 2D-LDA




% 3: Multi-class Linear SVM





% Useful Functions
%{

% show specific image
imshow(reshape(data_train(1,:,:),48,48));

%}
