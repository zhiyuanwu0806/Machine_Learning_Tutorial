function [updatedWeightMatrix, localGradient] = OutlayerBackPropogation(errors, d_activationFunction, ...
    weightMaxtrix, outputNodes, inputNodes, activationFunction, d_WeightMaxtrix, alpha, eta)
% 
% Description: Back-propogation for outlayer. M -> K. Total L layers.
% 
% errors:                   Errors. e_k^(L)(n)=d_k(n)-y_k^(L)(n)
%                               e_k^(L)(n). Kx1 column vector.
% d_activationFunction:     Derivative function of activation function. 
%                               d(\phi)/dx. 
% weightMatrix:             Current weight matrix.
%                               w_ki^(L)(n). KxM dimension.
% outputNodes:              Current output nodes without activation. 
%                           After activation, also as input nodes of the next level. 
%                               v_k^(L)(n). Kx1 column vector.
% inputNodes:               Current input nodes without activation. 
%                               v_i^(L-1)(n). Mx1 column vector. 
%                               y_i^(L-1)(n)=\phi(v_i^(L-1)(n))
% activationFunction:       Logistic function such as signoid, ReLU, etc.
%                               \phi.
% d_WeightMaxtrix:          Last difference of weight matrix.
%                               \Delta w_ki^(L)(n-1). KxM dimension.
% updatedWeightMatrix:      Updated weight matrix. 
%                               w_ki^(L)(n+1). KxM dimension.
% localGradient:            Current local gradient.
%                               \delta_k^(L)(n). Kx1 column vector.
%
% - Operation:
%   Momentum constant:        \alpha. absolute value in (0,1).
%   Learning rate:            \eta. small number. Like 0.1.
%   Local Gradient:
%     \delta_k^(L)(n)=(\phi)'(v_k^(L)(n))* e_k^(L)(n)
%   Weight matrix update:    
%     w_ki^(L)(n+1)= w_ki^(L)(n)+\alphac* [\Delta w_ki^(L)(n-1)]
%       +\eta* \delta_k^(L)(n)*y_i^(L-1)(n)
%     \Delta w_ki^(L)(n) = \alpha * [\Delta w_ki^(L)(n-1)] 
%       +\eta* \delta_k^(L)(n)*y_i^(L-1)(n)
%
% alpha=0.0;
% eta=1e-2;

% Local Gradient Calculation
if isa(d_activationFunction, 'function_handle') 
    localGradient = d_activationFunction(outputNodes).* errors;
end

% Weight Matrix Update
if isa(activationFunction, 'function_handle')
    updatedWeightMatrix = weightMaxtrix + alpha * d_WeightMaxtrix + eta* localGradient* activationFunction(inputNodes.');
end

end

