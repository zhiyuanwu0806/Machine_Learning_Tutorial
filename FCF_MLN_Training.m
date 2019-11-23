function FCF_MLN_Training()
% 
% Description: 
%   Train a fully connected forward multilayer network (FCF-MLN).
%   For simplicity, only two hidden layers.
% 

addpath ../MNIST
% Load Images
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% load('reducedImages.mat');
% Load Labes
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

numberSamples=length(labels);
% numberSamples=1000;

% Design multilayer network
inputNodeSize=784; hiddenLayer1=576; hiddenLayer2=96;
% inputNodeSize=196; hiddenLayer1=98; hiddenLayer2=49;
outputLayer=10;

% Dimension_MLN=[inputNodeSize hiddenLayer1 hiddenLayer2 outputLayer];
% number_Layer=length(Dimension_MLN);
% numberWeights=0;
%for ii=2:number_Layer
%	numberWeights=numberWeights+Dimension_MLN(ii-1)*Dimension_MLN(ii);
% end
% weightMatrix=ones(numberWeights,1);
% d_WeightMaxtrix=zeros(sie(weightMatrix));

% Initialize weight matrices
weightMatrix1=rand(hiddenLayer1,inputNodeSize)-0.5; % ones(hiddenLayer1,inputNodeSize)/inputNodeSize;% rand(hiddenLayer1,inputNodeSize)-0.5; % 
weightMatrix2=rand(hiddenLayer2,hiddenLayer1)-0.5; % ones(hiddenLayer2,hiddenLayer1)/hiddenLayer1; %
weightMatrix3=rand(outputLayer,hiddenLayer2)-0.5; % ones(outputLayer,hiddenLayer2)/hiddenLayer2; %
d_WeightMatrix3= zeros(size(weightMatrix3));
d_WeightMatrix2= zeros(size(weightMatrix2));
d_WeightMatrix1= zeros(size(weightMatrix1));

for it_o=1:100
    for nn=1:numberSamples
        
        % Load one sample
        currentLabel=labels(nn)
        % inputNodes=rd_images(:,nn);
        inputNodes=images(:,nn);
        
        % Set desired values according to the label
        desiredValues=zeros(outputLayer,1);
        desiredValues(currentLabel+1)=1;
        
        for it_i=1:10 %80
            
            % MLN Forward
            hidden1Nodes = weightMatrix1*inputNodes;
            hidden2Nodes=OneLayerForward(hidden1Nodes, @Sigmoid, weightMatrix2);
            outputNodes=OneLayerForward(hidden2Nodes, @Sigmoid, weightMatrix3);
            decisionNodes=Sigmoid(outputNodes);
            
            % Check errors
            errors=desiredValues-decisionNodes;
            sum_err=errors'*errors
            if sum_err<0.1
                break;
            end
            
            % MLN Back-Propogation
            [updatedWeightMatrix3, localGradient3] = OutlayerBackPropogation(errors, @d_Sigmoid, ...
                weightMatrix3, outputNodes, hidden2Nodes, @Sigmoid, d_WeightMatrix3, 0.0, 0.1);
            
            [updatedWeightMatrix2, localGradient2] = OneHiddenLayerBackPropogation( ...
                localGradient3, weightMatrix3, @d_Sigmoid, ...
                weightMatrix2, hidden2Nodes, hidden1Nodes, @Sigmoid, d_WeightMatrix2, 0.0, 0.1);
            
            [updatedWeightMatrix1, localGradient1] = OneHiddenLayerBackPropogation( ...
                localGradient2, weightMatrix2, @d_Sigmoid, ...
                weightMatrix1, hidden1Nodes, inputNodes, @PreFunc, d_WeightMatrix1, 0.0, 0.1);
            
            % Update weight matrices
            d_WeightMatrix3=updatedWeightMatrix3-weightMatrix3;
            weightMatrix3=updatedWeightMatrix3;
            d_WeightMatrix2=updatedWeightMatrix2-weightMatrix2;
            weightMatrix2=updatedWeightMatrix2;
            d_WeightMatrix1=updatedWeightMatrix1-weightMatrix1;
            weightMatrix1=updatedWeightMatrix1;
        end
    end
end

% Store trained weight matrics
save('fcf_WeightMatrix.mat','weightMatrix1','weightMatrix2','weightMatrix3');

end