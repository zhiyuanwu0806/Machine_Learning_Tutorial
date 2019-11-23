function [OutputData, Index_Pool]=Upsamp_ZeroPad(InputData, UpSampling_size, ZeroPadding_size)
% 
% This function performs forward up-sampling for convolutional neural network.
% InputData:        A column input vector
%                       M_in^2: (a square number) is the length of column vector InputData; 
% UpSampling_size:  A square number for up-sampling. 
%                   It shall be larger than "Depth^2". The depth is
%                   InputData column width.
% ZeroPadding_size: Zero padding size.
% OutputData:		A column output vector
% Index_Pool:		Pooling index for maximum or average sampling.

% For test only
test_ind=0;
if test_ind==1
    UpSampling_size=2^2; ZeroPadding_size=2;
    InputData=1:4^2*2;
    InputData=reshape(InputData,4^2,2);  
end

% Get output dimension
M_in=sqrt(length(InputData(:,1))); 
Depth=length(InputData(1,:));
K_sub=sqrt(UpSampling_size);
N_out=M_in*K_sub+2*ZeroPadding_size;
OutputData=zeros(N_out^2,1);
Index_Pool=OutputData;

% Output index matrix
p_ix=1:K_sub;
ix_start=p_ix+ZeroPadding_size+ZeroPadding_size*N_out;
for ii=1:M_in
    for jj=1:M_in
        ix_vec=(ii-1)*K_sub+(jj-1)*K_sub*N_out+ix_start;
        for kk=1:Depth
            ix_vec=ix_vec+(kk-1)*N_out+(kk-1);
            OutputData(ix_vec(1))=InputData((jj-1)*M_in+ii,kk);
            Index_Pool(ix_vec(1))=(jj-1)*M_in+ii+(kk-1)*M_in^2;
        end
    end
end

% for test only
if test_ind==1
    reshape(OutputData,N_out,N_out)
    reshape(Index_Pool,N_out,N_out)
end

end