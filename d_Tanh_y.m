function [dy] = d_Tanh_y(y)
% y=(exp(x)-exp(-x))/(exp(x)+exp(-x))
% x: column vector
% f'(x)=(1-f(x)^2);
% y=f(x): parameter to be differentiated
dy=(1-y.^2);

end

