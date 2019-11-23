function [y] = Tanh(x)
% y=tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))
% x: column vector

y=(exp(x)-exp(-x))./(exp(x)+exp(-x));

end

