function [updatedFilterBank, sumLocalGradientBank] = Conv_HiddenLayerBackPropogation( ...
    nextSumLocalGradientBank, IndexMatrix, d_activationFunction, ...
    FilterBank, outputNodeBank, inputNodeBank, d_FilterBank, alpha, eta)
% 
% Description: Back-propogation for one hidden layer.
% 
% [Input]
% nextSumLocalGradientBank: Sum of local gradient multiplying weight matrix propogated from the next level.
%                               \sum_k [\delta_k^(l+1)(n) * w_kj^(l+1)(n)]
% IndexMatrix:              Index matrix for convolution computation of one layer. 
%                               Row length=Output size x Column width=Filter length
% d_activationFunction:     Derivative function of activation function. 
%                               d(\phi)/dx. 
%                           activationFunction: Logistic function such as signoid, ReLU, etc.
% FilterBank:               Current filter bank. The column vector length shall be a square number
%                               w_ji^(l)(n).
% outputNodeBank:           Current output nodes without activation. 
%                           After activation, also as input nodes of the next level. 
%                               v_j^(l)(n). Mx1 column vector.
% inputNodeBank:            Current input node bank. 
%                               v_i^(l-1)(n). Nx1 column vector. 
%                               y_i^(l-1)(n)=\phi(v_i^(l-1)(n))
% d_FilterBank:             Last difference of filter.
%                               \Delta w_ij^(l)(n-1). MxN dimension.
% [Output]
% updatedFilterBank:        Updated filter bank. 
%                               w_ij^(l)(n+1). MxN dimension.
% sumLocalGradientBank:     Sum of local gradients multiplying filter.
%                               \sum_k [\delta_k^(l+1)(n) * w_kj^(l+1)(n)]
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

% Initialization
K_bank=length(outputNodeBank(1,:));
N_flt_2=length(FilterBank(:,1));
M_bank=length(inputNodeBank(1,:));
localGradientBank=zeros(size(outputNodeBank));
sumLocalGradientBank=zeros(size(inputNodeBank));
updatedFilterBank=zeros(size(FilterBank));

% Local Gradient Calculation
if isa(d_activationFunction, 'function_handle')
    for bb=1:K_bank
            localGradientBank(:,bb) = d_activationFunction(outputNodeBank(:,bb)).* nextSumLocalGradientBank(:,bb);
    end
end

% Filter Bank Update
for bb=1:K_bank
    for nn=1:N_flt_2
        up_temp= FilterBank(nn,bb)+ alpha * d_FilterBank(nn,bb);
        sum_temp=0;
        for cc=1:M_bank
            sum_temp=sum_temp+ localGradientBank(:,bb).'* (inputNodeBank(IndexMatrix(:,nn),cc));
        end
        updatedFilterBank(nn,bb)=up_temp+eta*sum_temp;
    end
end

% Calculate sum of local gradients multiplying filter weights
for bb=1:K_bank
    for nn=1:N_flt_2
        for cc=1:M_bank
            sumLocalGradientBank(IndexMatrix(:,nn),cc)=...
                sumLocalGradientBank(IndexMatrix(:,nn),cc)+FilterBank(nn,bb)*localGradientBank(:,bb);
        end
    end
end

end

