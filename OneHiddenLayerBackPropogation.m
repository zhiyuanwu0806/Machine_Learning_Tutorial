function [updatedWeightMatrix, localGradient] = OneHiddenLayerBackPropogation( ...
    nextLocalGradient, nextWeightMatrix, d_activationFunction, ...
    weightMaxtrix, outputNodes, inputNodes, activationFunction, d_weightMaxtrix, alpha, eta)
% 
% Description: Back-propogation for one hidden layer. N -> M -> K. 
% 
% [Input]
% nextLocalGradient:        Local gradients propogated from the next level.
%                               \delta_k^(l+1)(n). Kx1 column vector.
% nextWeightMatrix:         Weight matrix of the next level. 
%                               w_kj^(l+1)(n). KxM dimension.
% d_activationFunction:     Derivative function of activation function. 
%                               d(\phi)/dx. 
% weightMatrix:             Current weight matrix.
%                               w_ji^(l)(n). MxN dimension.
% outputNodes:              Current output nodes without activation. 
%                           After activation, also as input nodes of the next level. 
%                               v_j^(l)(n). Mx1 column vector.
% inputNodes:               Current input nodes without activation. 
%                               v_i^(l-1)(n). Nx1 column vector. 
%                               y_i^(l-1)(n)=\phi(v_i^(l-1)(n))
% activationFunction:       Logistic function such as signoid, ReLU, etc.
%                               \phi.
% d_weightMaxtrix:          Last difference of weight matrix.
%                               \Delta w_ij^(l)(n-1). MxN dimension.
% [Output]
% updatedWeightMatrix:      Updated weight matrix. 
%                               w_ij^(l)(n+1). MxN dimension.
% localGradient:            Current local gradient.
%                               \delta_j^(l)(n). Mx1 column vector.
%
% - Operation:
%   Momentum constant:        \alpha. absolute value in (0,1).
%   Learning rate:            \eta. small number. Like 0.1.
%   Local Gradient:
%     \delta_j^(l)(n)=(\phi)'(v_j^(l)(n)) * \sum_k [\delta_k^(l+1)(n) * w_kj^(l+1)(n)]
%   Weight matrix update:    
%     w_ij^(l)(n+1)= w_ij^(l)(n)+\alphac* [\Delta w_ij^(l)(n-1)]
%       +\eta* \delta_j^(l)(n)*y_i^(l-1)(n)
%     \Delta w_ij^(l)(n) = \alpha * [\Delta w_ij^(l)(n-1)] 
%       +\eta* \delta_j^(l)(n)*y_i^(l-1)(n)
%
% alpha=0.0;
% eta=1e-2;

% Local Gradient Calculation
if isa(d_activationFunction, 'function_handle') 
    localGradient = d_activationFunction(outputNodes).* (nextWeightMatrix.'*nextLocalGradient);
end

% Weight Matrix Update
if isa(activationFunction, 'function_handle')
    updatedWeightMatrix = weightMaxtrix + alpha * d_weightMaxtrix + eta* localGradient* activationFunction(inputNodes.');
end

end

