function ConvTrans_MLN()
% 
% Description: Make decision via trained convolutional multilayer network.
%

addpath ../MNIST
addpath ../Common
% Load Images & Labels
% Training samples
% images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');
% Test samples
% images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
% labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

% Design multilayer network
NoiseSize=4^2;
Conv_Layer_g0_size=8^2;  % 1st layer size
Conv_Layer_g0_Num=8;
Project_size=Conv_Layer_g0_size*Conv_Layer_g0_Num;
Conv_Filter_g1_Num=4;  % 1st convolutional layer number
UpSampling1_size=2^2;
ZeroPadding1_size=2;
Conv_Filter_g2_Num=1;  % 2nd convolutional layer number
UpSampling2_size=2^2;
ZeroPadding2_size=0;
Conv_Filter_g_Size=5^2;
Layer_g1_Width=sqrt(Conv_Layer_g0_size)*sqrt(UpSampling1_size)+ZeroPadding1_size*2;
Layer_g2_Width=Layer_g1_Width-sqrt(Conv_Filter_g_Size)+1;
% Initiallize weight matrix & filters.
WeightMatrix_g=2*rand(Project_size,NoiseSize)-1;
% trace(WeightMatrix_g*WeightMatrix_g')
FilterBank_g1=2*rand(Conv_Filter_g_Size,Conv_Filter_g1_Num)-1;
FilterBank_g2=2*rand(Conv_Filter_g_Size,Conv_Filter_g2_Num)-1;

% Generate noise data
NoiseVec=randn(NoiseSize,1);  % Generate noise vector
Layer_g0_Vec=Tanh(WeightMatrix_g*NoiseVec);  % Project
Layer_g0_Data=reshape(Layer_g0_Vec,Conv_Layer_g0_size,Conv_Layer_g0_Num);
% Layer 1
Layer_g1_Data=zeros(Layer_g1_Width^2,Conv_Filter_g1_Num);
Index_Pool_g1=zeros(size(Layer_g1_Data));
T_depth=Conv_Layer_g0_Num/Conv_Filter_g1_Num;
for ii=1:Conv_Filter_g1_Num
    ix_vec=(ii-1)*T_depth+(1:T_depth);
    [Layer_g1_Data(:,ii), Index_Pool_g1(:,ii)]=Upsamp_ZeroPad(...
        Layer_g0_Data(:,ix_vec), UpSampling1_size, ZeroPadding1_size);
end
[IndexMatrix_g1]=Conv_Index_Matrix(Layer_g1_Width,sqrt(Conv_Filter_g_Size),1);
OutputLayerBank_g1=zeros(Layer_g2_Width^2,Conv_Filter_g1_Num);
for ii=1:Conv_Filter_g1_Num
    temp_vec=Layer_g1_Data(:,ii);
    OutputLayerBank_g1(:,ii)=Tanh(temp_vec(IndexMatrix_g1)*FilterBank_g1(:,ii));  % Leaky_ReLU()/Tanh()/Sigmoid
end
% Layer 2
Layer_g2_Data=reshape(OutputLayerBank_g1,Layer_g2_Width^2*Conv_Filter_g1_Num,1);
T_width=Layer_g2_Width*sqrt(UpSampling2_size)+ZeroPadding2_size;
[IndexMatrix_g2]=Conv_Index_Matrix(T_width,sqrt(Conv_Filter_g_Size),1);
OutputLayerBank_g2=(Layer_g2_Data(IndexMatrix_g2)*FilterBank_g2);  % Leaky_ReLU()/Tanh()/Sigmoid
decisionNodes_g=Sigmoid(OutputLayerBank_g2);

test_ind=1;
if test_ind==1
    xx=reshape(decisionNodes_g,28,28);
    image(xx*255);
end

end