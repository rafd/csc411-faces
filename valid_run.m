ref_hist = imhist(reshape(data_train(1,:,:),imgRows,imgCols));

for i=1:size(data_valid,1)
    data_valid(i,:,:) = histeq(reshape(data_valid(i,:,:),imgRows,imgCols),ref_hist); 
end

fid = fopen('results.csv', 'w');


features_valid = zeros(size(data_valid,1), nVars);
data_valid = double(data_valid);
for i=1:size(data_valid,1)
    features_valid(i,:,:) = reshape((reshape(data_valid(i,:,:), imgRows, imgCols)*vec), nVars, 1);
end
[junk yhat_valid] = max([ones(size(data_valid,1),1) features_valid]*wSoftmax,[],2);

for i=1:size(data_valid,1)
    fprintf(fid, '%d\n', yhat_valid(i));
end
for i=1:835
  fprintf(fid, '%d\n', 1);
end

fclose(fid);
