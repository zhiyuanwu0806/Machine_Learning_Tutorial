function [outputNodes] = OneLayerForward(inputNodes, activationFunction, weightMaxtrix)
% 
% Description: outputNodes = weightMatrix*activationFunction(inputNodes).
%
% [Input]
% inputNodes:           Nx1 column vector without activation. v_i^(l-1)(n).
% activationFunction:   Logistic function such as signoid, ReLU, etc. 
% weightMatrix:         MxN matrix. w_ij^(l)(n).
% [Output]
% outputNodes:          Mx1 column vector without activation. v_j^(l)(n)
% - Operation:          
%                       y_i^(l-1)(n) = \phi(v_i^(l-1)(n)).
%                       v_j^(l)(n)= \sum_i [w_ij^(l)(n) * y_i^(l-1)(n)].
% 

if isa(activationFunction, 'function_handle')   % Verify that activationFunction is a function handle.
    outputNodes = weightMaxtrix*activationFunction(inputNodes);
end
end

