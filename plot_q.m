function plot_q(Q,I)
% NOTE THAT THE COMMENTS INDICATED EARLIER THAT 3 WAS LEFT AND 4 WAS DOWN,
% THIS WAS A MISTAKE. PLEASE NOTE THE CORRECTION.

% For each state, plots the action (direction) that will give the maximum
% Q-value.
% Q is a 10x10x4 matrix of state/action pairs, I is a 1x2 vector
% corresponding to the initial coordinates, where the origin (0,0) is the
% top-left of the grid, and (10,10) is the bottom right.
% The Q matrix is a 3D tensor of statesxstatesxactions where the indices of
% the actions correspond to the following directions:
% 1 - up
% 2 - right
% 3 - down
% 4 - left
c = gcf;
close(c);
hold on;
for i=1:size(Q,1)
    for j=1:size(Q,2)
        [q,d] = max(Q(i,j,:));
        u = 0;
        v = 0;
        q = q/10;
        hold on;
        if (d==1)
            v = q;
        elseif (d == 2)
            u = q;
        elseif (d == 3)
            v = -q;
        else
            u = -q;
        end
        quiver(j,size(Q,1)+1-i,u,v,'b','MaxHeadSize',0.8);
    end
end
plot(size(Q,1)+1-I(1),I(2),'r*');
hold off;