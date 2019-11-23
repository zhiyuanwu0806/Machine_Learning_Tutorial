function Conv_MLN()
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
images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

% Design multilayer network
inputNodeSize=28^2; %28x28=784
Conv_Filter_1_Size=5^2; % 1st convolutional filter size
Conv_Filter_1_Num=4;  % 1st convolutional filter number
Conv_Layer_1_Size=24^2; % 1st convolutional layer size associated one filter. =(28-5)+1
OutputBank1=zeros(Conv_Layer_1_Size,Conv_Filter_1_Num); % Output of 1st layer 
Subsampling_1_size=2^2; % pooling size of 1st sub-samping
Subsampling_Layer_1_size=Conv_Layer_1_Size/Subsampling_1_size; % 1st sub-sampling layer size. =12^2.
SubsamplingBank1=zeros(Subsampling_Layer_1_size,Conv_Filter_1_Num); % 1st subsampling layer output
IndexPoolingBank1=zeros(size(OutputBank1)); % 1st subsampling layer pooling results
Conv_Filter_2_Size=5^2; % 2nd convolutional filter size
Conv_Filter_2_Num=12;  % 2nd convolutional filter number
% Mapping between layer 1 and 2. 
Conv_Map=[1 1 1 2 2 2 3 3 3 4 4 4]; % length of 2nd convolutional filter number.
Conv_Layer_2_Size=8^2; % 2nd convolutional layer size associated one filter. =(12-5)+1
OutputBank2=zeros(Conv_Layer_2_Size,Conv_Filter_2_Num); % Output of 2nd layer 
Subsampling_2_size=2^2; % pooling size of 2nd sub-samping
Subsampling_Layer_2_size=Conv_Layer_2_Size/Subsampling_2_size;  % 2nd sub-sampling layer size. =4$2
SubsamplingBank2=zeros(Subsampling_Layer_2_size,Conv_Filter_2_Num); % 1st subsampling layer output
IndexPoolingBank2=zeros(size(OutputBank2)); % 2nd subsampling layer pooling results
outputLayer=10;  % Outlayer size

% Initialize Filter Banks
% Initialize filter banks & outlayer fully connected weight matrix
test_ind=0;
if test_ind==1
    FilterBank1=rand(Conv_Filter_1_Size,Conv_Filter_1_Num)-0.5;
    FilterBank2=rand(Conv_Filter_2_Size,Conv_Filter_2_Num)-0.5;
    weightMatrix=rand(outputLayer,Subsampling_Layer_2_size*Conv_Filter_2_Num)-0.5;
end

% Load trained filters & weights
load('conv_Filter&Weight.mat','FilterBank1','FilterBank2','weightMatrix');
% load('new_conv_Filter&Weight.mat','FilterBank1','FilterBank2','weightMatrix');

% Test range
N_start=1;
N_end=length(images(1,:));
%N_end=300;

err=0;
err_labels=[];
err_decision=[];
for nn_test=N_start:N_end
    % Load one sample
    currentLabel=labels(nn_test)
    inputNodes=images(:,nn_test);
    
    % Convolutional MLN Forward
    % Convolutional layer 1
    [IndexMatrix1]=Conv_Index_Matrix(sqrt(inputNodeSize),sqrt(Conv_Filter_1_Size),1);
    for bb=1:Conv_Filter_1_Num
        for ii=1:Conv_Layer_1_Size
            % without activation
            OutputBank1(ii,bb)= FilterBank1(:,bb).'*inputNodes(IndexMatrix1(ii,:));
        end
    end
    % Sub-sampling layer 1
    Type=0; % Max
    for bb=1:Conv_Filter_1_Num
        [SubsamplingBank1(:,bb),IndexPoolingBank1(:,bb)]=...
            Conv_Subsampling(OutputBank1(:,bb),Subsampling_1_size, @ReLU, Type);
    end
    % Convolutional layer 2
    [IndexMatrix2]=Conv_Index_Matrix(sqrt(Subsampling_Layer_1_size),sqrt(Conv_Filter_2_Size),1);
    for bb=1:Conv_Filter_2_Num
        for ii=1:Conv_Layer_2_Size
            OutputBank2(ii,bb)= FilterBank2(:,bb).'*SubsamplingBank1(IndexMatrix2(ii,:),Conv_Map(bb));
        end
    end
    % Sub-sampling layer 2
    Type=0; % Max
    for bb=1:Conv_Filter_2_Num
        [SubsamplingBank2(:,bb),IndexPoolingBank2(:,bb)]=...
            Conv_Subsampling(OutputBank2(:,bb),Subsampling_2_size, @ReLU, Type);
    end
    % Outlayer 
    outputNodes=weightMatrix*reshape(SubsamplingBank2,Subsampling_Layer_2_size*Conv_Filter_2_Num,1); 
    decisionNodes=Sigmoid(outputNodes);
    
    [mm ix]=max(decisionNodes);
    % Show the decision
    ix-1
    if ((ix-1)~=currentLabel)
        err=err+1;
        err_labels=[err_labels nn_test];
        err_decision=[err_decision ix-1];
    end
end
Identification_rate=1-err/(N_end-N_start+1)
save('conv_result.mat','Identification_rate','err_labels','err_decision');
% save('new_conv_result.mat','Identification_rate','err_labels','err_decision');

end