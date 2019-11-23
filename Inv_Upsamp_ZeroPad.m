function [InputDataBank]=Inv_Upsamp_ZeroPad(OutputData, Depth, UpSampling_size, ZeroPadding_size, Index_Pool)
% 
% This function performs forward up-sampling for convolutional neural network.
% OutputData:		A column vector after zero-pading and up-sampling.
%						With a square length(N_out=M_in*K_sub+2*ZeroPadding_size)
% Depth:			Input data bank depth
% UpSampling_size:  A square number for up-sampling (K_sub^2).
% ZeroPadding_size: Zero padding size.
% Index_Pool:		Pooling index for maximum or average up-sampling.
% InputDataBank:		A column vector before zero-pading and up-sampling.
%						with a square number length (M_in^2) 
%						M_in=(N_out-2*ZeroPadding_size)/K_sub.

% For test only
test_ind=0;
if test_ind==1
    OutputData=(1:8^2)';  
    UpSampling_size=2^2; 
    ZeroPadding_size=2;
    Index_Pool=ones(8^2,1);
	Depth=2;
end

% Get output dimension
N_out=sqrt(length(OutputData)); 
K_sub=sqrt(UpSampling_size);
M_in=(N_out-2*ZeroPadding_size)/K_sub;
InputDataBank=zeros(M_in^2,Depth);

% Remove irrelevant
OutputData=OutputData.*Index_Pool;
% Input index matrix
p_ix=1:K_sub;
ix_start=p_ix+ZeroPadding_size+ZeroPadding_size*N_out;
for ii=1:M_in
    for jj=1:M_in
        ix_vec=(ii-1)*K_sub+(jj-1)*K_sub*N_out+ix_start;
        for kk=1:Depth
            ix_vec=ix_vec+(kk-1)*N_out+(kk-1);
            InputDataBank((jj-1)*M_in+ii,kk)=OutputData(ix_vec(1));
        end
    end
end

% for test only
if test_ind==1
    reshape(OutputData,N_out,N_out)
    reshape(InputDataBank,M_in,M_in,Depth)
end

end