clc; 
clear all; 
 
num_e_vecs_to_use = 5;

%load face database 
load('TFD_hw4_train.mat')
facedatabase = double(data_train); 

% Massage the data into the form desired by the 2D LDA function
nclass = 7; 
nsmpaleeachclass = 417; 
height = 48; 
width = 48; 
 
for i=1:neachtrain*nclass 
    trainSample{i}=reshape(facedatabase(i,:,:), height, width); 
end 
 
for i=1:neachtrain*nclass 
    testSample{i}=reshape(facedatabase(i,:,:), height, width); 
end 

[vec, val] = tdfda(trainSample, nclass); 

% Decide which eigenvectors to use
