function [LocalGradient]=Conv_Upsampling(nextLocalGradient, IndexPooling, ...
    Sampling_size)
% 
% This function performs up-sampling (inverse-pooling) back propogation for convolutional neural
% network.
% nextLocalGradient:    Local gradient after pooling. A column vector with a length of square number
% IndexPooling:         Weighting for up-sampling according to Type. It is the same size as nextLocalGradient.
% Type:                 Maximum (=0) or Average (=1). It is the same size as nextLocalGradient. 
% Sampling_Size:        A square number for sub-sampling (pooling)
% LocalGradient:        Up-sampling local gradient output.

% For test only
test_ind=0;
if test_ind==1
    nextLocalGradient=(1:2^2)'; Sampling_size=2^2;
end

% Get dimension
L_out=sqrt(length(nextLocalGradient));
K_sub=sqrt(Sampling_size);
M_in=(L_out*K_sub);
LocalGradient=zeros(M_in^2,1);
vector_ix=zeros(1,Sampling_size);

% For test only
if test_ind==1
    IndexPooling=ones(size(LocalGradient));%/Sampling_size;
    Samp_IndexMatrix=zeros(L_out^2,Sampling_size);
end

% Output local gradients
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
        % Calculation for local gradients
        LocalGradient(vector_ix)=nextLocalGradient((ii-1)*L_out+jj);
    end
end
% Final outputs
LocalGradient=LocalGradient.*IndexPooling;

end