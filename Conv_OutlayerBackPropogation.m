function [updatedWeightMatrix, sumLocalGradient] = Conv_OutlayerBackPropogation(errors, d_activationFunction, ...
    weightMatrix, outputNodes, inputNodes, d_WeightMatrix, alpha, eta)
% 
% Description: Back-propogation for convolutional neural network outlayer. 
% Channel number {D_ch) x Output dimension (L_out^2)  M -> K. Total L layers. 
% Please pay attention that CNN outlayer is fully connected NN.
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
% inputNodes:               Current input nodes. 
%                               y_j^(L-1)(n)=\phi(v_j^(L-1)(n))
% d_WeightMatrix:          Last difference of weight matrix.
%                               \Delta w_ki^(L)(n-1). KxM dimension.
% updatedWeightMatrix:      Updated weight matrix. 
%                               w_ki^(L)(n+1). KxM dimension.
% sumLocalGradient:         Sum of local gradient multiplying weight matrix for upper level. 
%                               \sum_k (delta_k^(L)(n)*w_ki^(L)(n)). 
%                               \delta_k^(L)(n). Kx1 column vector.
%
% - Operation:
%   Momentum constant:        \alpha. absolute value in (0,1).
%   Learning rate:            \eta. small number. Like 0.1.
%   Local Gradient:
%     \delta_k^(L)(n)=(\phi)'(v_k^(L)(n))* e_k^(L)(n)
%   Weight matrix update:    
%     w_ki^(L)(n+1)= w_ki^(L)(n)+\alpha* [\Delta w_ki^(L)(n-1)]
%       +\eta* \delta_k^(L)(n)*y_i^(L-1)(n)
%     \Delta w_ki^(L)(n) = \alpha * [\Delta w_ki^(L)(n-1)] 
%       +\eta* \delta_k^(L)(n)*y_i^(L-1)(n)
%   Sum of local gradients multiplying weight matrix:
%      \sum_k (delta_k^(L)(n)*w_ki^(L)(n)).
% alpha=0.0;
% eta=1e-2;

% Local Gradient Calculation
if isa(d_activationFunction, 'function_handle') 
    localGradient = d_activationFunction(outputNodes).* errors;
end

% Weight Matrix Update
updatedWeightMatrix = weightMatrix + alpha * d_WeightMatrix + eta* localGradient* inputNodes.';

% Calculate sum of local gradient multiplying weight matrix for upper level
sumLocalGradient= weightMatrix.'*localGradient;

end

