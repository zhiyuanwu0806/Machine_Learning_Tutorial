function [y] = Sigmoid(x)
% y=1/(1+exp(-alf*x))
% x: column vector
alf=1;
small_value=1e-60;
y=1./(1+exp(-alf*x)+small_value);

end

