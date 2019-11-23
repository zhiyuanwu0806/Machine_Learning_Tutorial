function [y] = Leaky_ReLU(x)
% Leaky recified unit function
% x: column vector
% a constant larger than 1
% y=x if x>=0; y=x/a; if x<0.

a=10;
y=x;
ix=find(x<0);
y(ix)=x(ix)/a;

end

