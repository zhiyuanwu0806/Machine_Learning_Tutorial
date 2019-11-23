function [y] = d_Sigmoid(x)
% y=d(1/(1+exp(-alf*x)))/dx
% x: column vector
alf=1;
small_value=1e-60;
y=alf*exp(-alf*x)./((1+exp(-alf*x)).^2+small_value);

end

