function [y] = d_Leaky_ReLU(x)
% Derivative of Leaky recified unit function
% x: column vector
% a constant larger than 1
% y=x if x>=0; y=x; if x<0.
% dy/dx=1 if x>0; dy/dx=1/a if x<0; dy/dx=(1+1/a)/2 if x=0.
 
a=10;
y=ones(size(x));
ix=find(x<0);
y(ix)=1/a;
ix0=find(x==0);
y(ix0)=(1+1/a)/2;

end


