function[y] = oneOfKtoNum(targets)
% Converts a matrix of targets where each row is a 
% 1 of K vector corresponding to the class of that example
% into a single vector where each row contains an integer from 1 to K
% corresponding to the class of that example.
[junk,y] = max(targets,[],2);