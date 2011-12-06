function[X] = sample_1_of_k(p)
% Given a matrix p with normalized rows, samples a 1 of k matrix where each
% row of the matrix is a 1 of k vector that has been sampled from the
% distribution of that row.
cp = cumsum(p,2);
[junk,x] = max(repmat(rand(size(p,1),1),1,size(p,2))<cp,[],2);
X = num2OneOfK(x,size(p,2));