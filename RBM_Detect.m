function RBM_Detect()
% This function to test restricted Boltzmann machine (RBM).

addpath ../MNIST
% Load Images & Labels
% Training samples
% images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');
% Test samples
images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

% numberSamples=length(labels);
numberSamples=1;

% Design restricted Boltzmann Machine
inputNodeSize=784;  % Visible nodes
% inputNodeSize=196;  % Visible nodes
outputLayer=576; % Hidden nodes

% Load weight matrices
load('rBM_CD_weightMaxtrix.mat','weightMatrix','bias4Hidden','bias4Visible');

n_test=[1 12];
for nn=n_test
    % Load one sample
    % nn
    currentLabel=labels(nn)
     %if currentLabel == 5
        inputNodes=images(:,nn);
        
        h_out=(weightMatrix*inputNodes+bias4Hidden)';
        Sigmoid(h_out')'
    %end
end

end

