function [dy] = d_Sigm(y)
% f(x)=sigmoid(x)=1/(1+exp(-x))
% f'(x)=f(x)*(1-f(x));
% y=f(x): parameter to be differentiated
dy=y.*(1-y);

end

