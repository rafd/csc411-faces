[y,x] = find(targets_train');

sort_index = sortrows([x,y],2);

sorted_labels = sort_index(:,2);

sorted_faces = zeros(size(data_train));

for i=1:size(data_train,1)
    sorted_faces(i,:,:) = data_train(sort_index(i,1),:,:);
end