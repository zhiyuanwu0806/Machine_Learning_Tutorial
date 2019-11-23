function ForwardMultiLayerPerceptron()
% 
% Description: Make decision via trained forward multilayer network.
%

addpath ../MNIST
% Load Images & Labels
% Training samples
% images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% load('reducedImages.mat'); % Reduced size images
% labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');
% Test samples
images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

% Load my samples
% load('Pattern.mat','Pattern');
% images=Pattern;

% Design multilayer network
% inputNodeSize=784; hiddenLayer1=576; hiddenLayer2=96; 
% If "reducedImages", then
% inputNodeSize=196; hiddenLayer1=98; hiddenLayer2=49; 
% outputLayer=10;

% Load MLP Weight Matrices
load('fcf_WeightMatrix.mat','weightMatrix1','weightMatrix2','weightMatrix3');
% load('pcf_WeightMatrix.mat','weightMatrix1','weightMatrix2','weightMatrix3');

% Special samples to be test
% load('err_common.mat');
% images=images(:,err_common_labels);
% labels=labels(err_common_labels);

% Test range
N_start=1;
N_end=length(images(1,:));
% N_end=2000;

err=0;
err_labels=[];
err_decision=[];
for nn_test=N_start:N_end
    % Load one sample
    currentLabel=labels(nn_test)
    %inputNodes=rd_images(:,nn_test);
    inputNodes=images(:,nn_test);
    
    % img_2d=reshape(inputNodes,28,28); image(img_2d*255);
    
    % MLN Forward
    hidden1Nodes = weightMatrix1*inputNodes;
    hidden2Nodes=OneLayerForward(hidden1Nodes, @Sigmoid, weightMatrix2);
    outputNodes=OneLayerForward(hidden2Nodes, @Sigmoid, weightMatrix3);
    decisionNodes=Sigmoid(outputNodes)'
    
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
save('fcf_result.mat','Identification_rate','err_labels','err_decision');
% save('pcf_result.mat','Identification_rate','err_labels');

end