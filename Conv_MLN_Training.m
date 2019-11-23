function Conv_MLN_Training()
% 
% Description: 
%   Train a convoluitonal multilayer network (Conv-MLN).
%   For simplicity, only two hidden convolution+pooling layers.
% 

addpath ../MNIST
addpath ../Common
% Load Training Images & Labels
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

numberSamples=length(labels);
% numberSamples=1000;

% Design multilayer network
inputNodeSize=28^2; %28x28=784
Conv_Filter_1_Size=5^2; % 1st convolutional filter size
Conv_Filter_1_Num=4;  % 1st convolutional filter number
Conv_Layer_1_Size=24^2; % 1st convolutional layer size associated one filter. =(28-5)+1
OutputBank1=zeros(Conv_Layer_1_Size,Conv_Filter_1_Num); % Output of 1st layer without activation
Subsampling_1_size=2^2; % pooling size of 1st sub-samping
Subsampling_Layer_1_size=Conv_Layer_1_Size/Subsampling_1_size; % 1st sub-sampling layer size. =12^2.
SubsamplingBank1=zeros(Subsampling_Layer_1_size,Conv_Filter_1_Num); % 1st subsampling layer output
IndexPoolingBank1=zeros(size(OutputBank1)); % 1st subsampling layer pooling results
Conv_Filter_2_Size=5^2; % 2nd convolutional filter size
Conv_Filter_2_Num=12;  % 2nd convolutional filter number
% Mapping between layer 1 and 2. 
N_expansion=Conv_Filter_2_Num/Conv_Filter_1_Num;
Conv_Map=kron(1:Conv_Filter_1_Num,ones(1,N_expansion)); % length of 2nd convolutional filter number.
Conv_Layer_2_Size=8^2; % 2nd convolutional layer size associated one filter. =(12-5)+1
OutputBank2=zeros(Conv_Layer_2_Size,Conv_Filter_2_Num); % Output of 2nd layer without activation
Subsampling_2_size=2^2; % pooling size of 2nd sub-samping
Subsampling_Layer_2_size=Conv_Layer_2_Size/Subsampling_2_size;  % 2nd sub-sampling layer size. =4^2
SubsamplingBank2=zeros(Subsampling_Layer_2_size,Conv_Filter_2_Num); % 1st subsampling layer output
IndexPoolingBank2=zeros(size(OutputBank2)); % 2nd subsampling layer pooling results
outputLayer=10;  % Outlayer size

% Initialize filter banks & outlayer fully connected weight matrix
FilterBank1=rand(Conv_Filter_1_Size,Conv_Filter_1_Num)-0.5; 
FilterBank2=rand(Conv_Filter_2_Size,Conv_Filter_2_Num)-0.5; 
weightMatrix=rand(outputLayer,Subsampling_Layer_2_size*Conv_Filter_2_Num)-0.5;

% Differences
d_FilterBank1= zeros(size(FilterBank1));
d_FilterBank2= zeros(size(FilterBank2));
d_weightMatrix=zeros(size(weightMatrix));

for it_o=1:10 %100
    for nn=1:numberSamples
        
        % Load one sample
        currentLabel=labels(nn)
        inputNodes=images(:,nn);
        
        % Set desired values according to the label
        desiredValues=zeros(outputLayer,1);
        desiredValues(currentLabel+1)=1;
        
        for it_i=1:80; %80
            
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
                % Activation done and sub-sampling
            end
            % Convolutional layer 2
            [IndexMatrix2]=Conv_Index_Matrix(sqrt(Subsampling_Layer_1_size),sqrt(Conv_Filter_2_Size),1);
            for bb=1:Conv_Filter_2_Num
                for ii=1:Conv_Layer_2_Size
                    % without activation
                    OutputBank2(ii,bb)= FilterBank2(:,bb).'*SubsamplingBank1(IndexMatrix2(ii,:),Conv_Map(bb));
                end
            end
            % Sub-sampling layer 2
            Type=0; % Max
            for bb=1:Conv_Filter_2_Num
                [SubsamplingBank2(:,bb),IndexPoolingBank2(:,bb)]=...
                    Conv_Subsampling(OutputBank2(:,bb),Subsampling_2_size, @ReLU, Type);
                % Activation done and sub-sampling
            end
            % Outlayer
            SubsamplingNodes2=reshape(SubsamplingBank2,Subsampling_Layer_2_size*Conv_Filter_2_Num,1);
            outputNodes=weightMatrix*SubsamplingNodes2;
            decisionNodes=Sigmoid(outputNodes);
            
            % Check errors
            errors=desiredValues-decisionNodes;
            sum_err=abs(errors'*errors);
            if sum_err<0.01
                break;
            end
            
            % Convoltional MLN Back-Propogation
            % Outlayer BP
            [updatedWeightMatrix, sumLocalGradient] = Conv_OutlayerBackPropogation(...
                errors, @d_Sigmoid, weightMatrix, outputNodes, SubsamplingNodes2, ...
                d_weightMatrix, 0.0, 0.1);
            sumLocalGradientBank=reshape(sumLocalGradient,Subsampling_Layer_2_size,Conv_Filter_2_Num);
            % Up-sampling layer 2
            sumUpLocalGradientBank=zeros(size(OutputBank2));
            for bb=1:Conv_Filter_2_Num
                [sumUpLocalGradientBank(:,bb)]=Conv_Upsampling(...
                    sumLocalGradientBank(:,bb), IndexPoolingBank2(:,bb), Subsampling_2_size);
            end
            % Convolutional layer 2 BP
            updatedFilterBank2=zeros(size(FilterBank2));
            sumLocalGradientBank2=zeros(size(SubsamplingBank1));
            grp_num=Conv_Filter_2_Num/Conv_Filter_1_Num;
            for bb=1:Conv_Filter_1_Num
                ix_temp=(1:grp_num)+(bb-1)*grp_num;
                [updatedFilterBank2(:,ix_temp), sumLocalGradientBank2(:,bb)] = ...
                    Conv_HiddenLayerBackPropogation(sumUpLocalGradientBank(:,ix_temp), ...
                    IndexMatrix2, @d_ReLU, FilterBank2(:,ix_temp), OutputBank2(:,ix_temp), ...
                    SubsamplingBank1(:,bb), d_FilterBank2(:,ix_temp), 0.0, 0.05);
            end
            % Up-sampling layer 1
            sumUpLocalGradientBank1=zeros(size(OutputBank1));
            for bb=1:Conv_Filter_1_Num
                [sumUpLocalGradientBank1(:,bb)]=Conv_Upsampling(...
                    sumLocalGradientBank2(:,bb), IndexPoolingBank1(:,bb), Subsampling_1_size);
            end
            % Convolutional layer 1 BP
            [updatedFilterBank1, sumLocalGradientBank1] = Conv_HiddenLayerBackPropogation( ...
                sumUpLocalGradientBank1, IndexMatrix1, @d_ReLU, ...
                FilterBank1, OutputBank1, inputNodes, d_FilterBank1, 0.0, 0.05);
            
            % Update filters & weights
            d_weightMatrix=updatedWeightMatrix-weightMatrix;
            weightMatrix=updatedWeightMatrix;
            d_FilterBank2=updatedFilterBank2-FilterBank2;
            FilterBank2=updatedFilterBank2;
            d_FilterBank1=updatedFilterBank1-FilterBank1;
            FilterBank1=updatedFilterBank1;
        end
    end
end

% Store trained weight matrics
save('conv_Filter&Weight.mat','FilterBank1','FilterBank2','weightMatrix');

end