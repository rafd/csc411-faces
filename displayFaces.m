function h=displayFaces(A, numcols, figstart)
% Usage: displayFaces(data)
% Produces a figure displaying faces from the Toronto Faces Dataset.
% Written by: Honglak Lee
% Modified by: Kevin Swersky
[num_images,num_cols,num_rows] = size(A);
A = reshape(A,num_images,num_cols*num_rows)';
warning off all

if exist('figstart', 'var') && ~isempty(figstart), figure(figstart); end

[L M]=size(A);
if ~exist('numcols', 'var')
    numcols = ceil(sqrt(L));
    while mod(L, numcols), numcols= numcols+1; end
end
ysz = numcols;
xsz = ceil(L/ysz);

m=floor(sqrt(M*ysz/xsz));
n=ceil(M/m);

colormap(gray);

buf=1;
array=-ones(buf+m*(xsz+buf),buf+n*(ysz+buf));

k=1;
for i=1:m
    for j=1:n
        if k>M continue; end
        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])=...
            reshape(A(:,k),xsz,ysz);
        k=k+1;
    end
end

if isreal(array)
    h=imagesc(array);
else
    h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
end;
axis image off;

drawnow;

warning on all;
