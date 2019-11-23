function [y] = d_ReLU(x)
% y=d(max(x,0))/dx
% x: column vector
% y=max(x,0);
 
y=zeros(size(x));
 
ix_g=find(x>0);
y(ix_g)=1;
ix_l=find(x==0);
y(ix_l)=0.5;
 
end


