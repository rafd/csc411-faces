function [vec, val] = tdfda(sample, nClass, labels) 
 
nSample = size(sample, 2); 
[height, width] = size(sample{1});
nSamplePerClass = nSample/nClass; 
 
%
meanSample = zeros(height, width); 
for i=1:nSample 
    meanSample = meanSample + sample{i}; 
end 
meanSample = meanSample/nSample; 
%------------------------------------------- 
% 
last_class = labels(1);
class = 1;
meanClass{class} = zeros(height, width);
classTotal{class} = 0;
for i=1:nSample
	if (last_class == labels(i))
		meanClass{class} = meanClass{class} + sample{i};
		classTotal{class} = classTotal{class} + 1;
	else
		meanClass{class} = meanClass{class}/classTotal{class};
		last_class = labels(i);
		class = class + 1;
		classTotal{class} = 1;
		meanClass{class} = zeros(height, width);
	end
end
%-------------------------------------------------- 
meanClass{class} = meanClass{class} / classTotal{class};

Gb = zeros(width, width); 
for i=1:nClass 
    Gb = Gb + classTotal{i}*(meanClass{i}-meanSample)'*(meanClass{i}-meanSample); 
end 
Gb = Gb/nSample; 
 
Gw = zeros(width, width);
for i=1:nSample
	class = labels(i);
	Gw = Gw + (sample{i} - meanClass{class})' * (sample{i} - meanClass{class});
end
Gw = Gw/nSample;
 
Gt = zeros(width, width); 
for i=1:nSample 
    Gt = Gt + (sample{i}-meanSample)'*(sample{i}-meanSample); 
end 
Gt = Gt/nSample; 
 
[vec, val] = eig(Gb,Gw); 
%[vec, val] = eig(inv(Gw)*Gb);%两式作用相同 
[vec, val] = sortem(vec, val); 
 
val = diag(val); 
for i=1:size(vec,2) 
    vec(:,i)=vec(:,i)/norm(vec(:,i)); 
end