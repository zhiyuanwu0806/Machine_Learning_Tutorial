function [Conv_IndexMatrix]=Conv_Index_Matrix(M_in,N_flt,Step)
% Description: This function is to compute the index matrix for convolution computation of one layer. 
% M_in^2:           (a square number) is the length of column vector InputData; 
% N_flt^2:          (a square number) is the length of column vector Filter (or kernel).
%                       N_flt mus be smaller than M_in.
% Step:             Moving step.
% Conv_IndexMatrix: Index matrix for convolution.
% 
% For column vector=[1;2;3;4]. The corresponding square matrix is [1 3; 2 4].
% OutputData is also a column vector with a length of square number.
% L_out^2 (a square number) is the length of column vector OutputData. 
% L_out is equal to fix((M_in-N_flt)/Step)+1.

% For test only
test_ind=0;
if test_ind==1
    InputData=(1:5^2)';  Filter=ones(2^2,1);
    Step=1; M_in=sqrt(length(InputData)); N_flt=sqrt(length(Filter));
end

% Get dimension
L_out=fix((M_in-N_flt)/Step)+1;
Conv_IndexMatrix=zeros(L_out^2,N_flt^2);
vector_ix=zeros(1,N_flt^2);

p_ix=1:N_flt;
for ii=1:L_out
    for jj=1:L_out
        ix_start=(ii-1)*Step+(jj-1)*Step*M_in+p_ix;
        for kk=1:N_flt
            vector_ix(p_ix+(kk-1)*N_flt)=ix_start+(kk-1)*M_in;
        end
        Conv_IndexMatrix((ii-1)*L_out+jj,:)=vector_ix;

    end
end

% for test only
if test_ind==1
    OutputData=InputData(Conv_IndexMatrix)*Filter
end

end

