function [OutputDecisions,IndexPooling]=Conv_Subsampling(InputData,...
    Sampling_size, activationFunction, Type)
%
% This function performs sub-sampling (pooling) for convolutional neural network.
% InputData:                Input data without activation yet. 
%                           A column vector with a length of square number.
% Sampling_Size:            A square number for sub-sampling (pooling).
% activationFunction:       Logistic function such as signoid, ReLU, etc.
%                               \phi.
% Type:                     Maximum (=0) or Average (=1).
% OutputDecisions:          Sub-sampling output.
% IndexPooling:             Record pooling indices.

% For test only
test_ind=0;
if test_ind==1
    InputData=(1:6^2)'; Sampling_size=2^2; Type=0; activationFunction=@ReLU;
end

% Get dimension
M_in=sqrt(length(InputData));
K_sub=sqrt(Sampling_size);
L_out=(M_in/K_sub);
OutputDecisions=zeros(L_out^2,1);
IndexPooling=zeros(size(InputData));
vector_ix=zeros(1,Sampling_size);
% For test only
if test_ind==1
    Samp_IndexMatrix=zeros(L_out^2,Sampling_size);
end

% Activation
if isa(activationFunction, 'function_handle')
    actInputData=activationFunction(InputData);
end

% Output Decisions
for ii=1:L_out
    for jj=1:L_out
        p_ix=1:K_sub;
        ix_start=(ii-1)*K_sub+(jj-1)*K_sub*M_in+(1:K_sub);
        for kk=1:K_sub
            vector_ix(p_ix+(kk-1)*K_sub)=ix_start+(kk-1)*M_in;
        end
        % For test only
        if test_ind==1
            Samp_IndexMatrix((ii-1)*L_out+jj,:)=vector_ix;
        end

        if Type==0  % Maximum in pooling 
            [v_max,ix_max]=max(actInputData(vector_ix));
            OutputDecisions((ii-1)*L_out+jj)=v_max;
            IndexPooling(vector_ix(ix_max))=1;  % Keep indices for upsampling
        else  % Average in pooling
            OutputDecisions((ii-1)*L_out+jj)=mean(actInputData(vector_ix));
            IndexPooling(vector_ix)=1/Sampling_size; % For upsampling weighting
        end
    end
end

end