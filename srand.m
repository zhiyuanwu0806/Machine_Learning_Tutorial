function [Ix_il,Ix_deil]=srand(N_block,Min_Dis)
%%% S-random interleaver Design --by X.F. Wang
%%  -----Inputs--- 
%%% N_block: block length
%%% Min_Dis: minimal distance (Min_Dis<sqrt(N_block/2))
%%% Output Ix_il (Ix_deil): Index vector of the output of (de)interleaver. 
%%%%    for a input vector x, x(Ix_il) will be the output of the interleaver
%%%%   and y(Ix_deil) will be the output of the corresponding deinterleaver

% Condition Check
if Min_Dis >= sqrt(N_block/2)
    fprintf(2,'The minimal distance should be less than sqrt(N_block/2)');
    %Default value of minimal distance
    Min_Dis=fix(sqrt(N_block/2))
end

rand('state',sum(100*clock));
Ix_il=floor(N_block*rand(1));
Ix_vector=0:N_block-1;
Ix_vector=[Ix_vector(1:Ix_il) Ix_vector(Ix_il+2:N_block)];
Ix_len=length(Ix_il);
Ix_deil=[];
while Ix_len < N_block-1
    x=floor((N_block-Ix_len)*rand(1));
    y=Ix_vector(x+1);
    while min(abs(y-Ix_il(max(Ix_len-Min_Dis+2,1):Ix_len))) < Min_Dis
        x=floor((N_block-Ix_len)*rand(1));
        y=Ix_vector(x+1);
    end
    Ix_il=[Ix_il y];
    Ix_vector=[Ix_vector(1:x) Ix_vector(x+2:N_block-Ix_len)];
    Ix_len=Ix_len+1;
end
Ix_il=[Ix_il Ix_vector];
Ix_il=Ix_il(:)+1;
[temp Ix_deil]=sort(Ix_il);
% filename=sprintf('srand_N%d.mat',N_block);
% save(filename,'Ix_il','Ix_deil');

