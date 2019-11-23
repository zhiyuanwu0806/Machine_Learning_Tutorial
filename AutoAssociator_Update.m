function AutoAssociator_Update()
% This function to update Auto-Associator (AA).

% Some parameters
alf=0.1;
elp=0.05;

addpath ../MNIST
% Load Images & Labels
% Training samples
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

numberSamples=length(labels);
% numberSamples=10000;

% Design multilayer network for AA.
inputNodeSize=784; hiddenLayer1=576; 
hiddenLayer2=192; hiddenLayer3=64;
outputLayer=16;

% Initialize weight matrices
% Layer 1
weightMatrix1=alf*(rand(hiddenLayer1,inputNodeSize)-0.5);
bias4Hidden1=elp*(rand(hiddenLayer1,1)-0.5);
bias4Visible1=elp*(rand(inputNodeSize,1)-0.5);
% Layer 2
weightMatrix2=alf*(rand(hiddenLayer2,hiddenLayer1)-0.5);
bias4Hidden2=elp*(rand(hiddenLayer2,1)-0.5);
bias4Visible2=elp*(rand(hiddenLayer1,1)-0.5);
% Layer 3
weightMatrix3=alf*(rand(hiddenLayer3,hiddenLayer2)-0.5);
bias4Hidden3=elp*(rand(hiddenLayer3,1)-0.5);
bias4Visible3=elp*(rand(hiddenLayer2,1)-0.5);
% Outlayer
weightMatrix4=alf*(rand(outputLayer,hiddenLayer3)-0.5);
bias4Hidden4=elp*(rand(outputLayer,1)-0.5);
bias4Visible4=elp*(rand(hiddenLayer3,1)-0.5);

N_iteration=20; % Internal iteration number
rand('state',0);

counter=0;

for itr_o=1:5
    for nn=1:numberSamples
        % Load one sample
        currentLabel=labels(nn);
        inputNodes=images(:,nn);
        
        % Just a counter to show current progress
        counter=counter+1
        
        % Layer 1
        [HiddenNodes1,weightMatrix1,bias4Hidden1,bias4Visible1]=...
            AA_OneHiddenLayer(inputNodes,weightMatrix1,...
            bias4Hidden1,bias4Visible1,N_iteration);
        save('AA_weightMaxtrix1.mat','weightMatrix1','bias4Hidden1','bias4Visible1');
        
        % Layer 2
        [HiddenNodes2,weightMatrix2,bias4Hidden2,bias4Visible2]=...
            AA_OneHiddenLayer(HiddenNodes1,weightMatrix2,...
            bias4Hidden2,bias4Visible2,N_iteration);
        save('AA_weightMaxtrix2.mat','weightMatrix2','bias4Hidden2','bias4Visible2');

        % Layer 3
        [HiddenNodes3,weightMatrix3,bias4Hidden3,bias4Visible3]=...
            AA_OneHiddenLayer(HiddenNodes2,weightMatrix3,...
            bias4Hidden3,bias4Visible3,N_iteration);
        save('AA_weightMaxtrix3.mat','weightMatrix3','bias4Hidden3','bias4Visible3');

        % OutLayer
        [OutputNodes,weightMatrix4,bias4Hidden4,bias4Visible4]=...
            AA_OneHiddenLayer(HiddenNodes3,weightMatrix4,...
            bias4Hidden4,bias4Visible4,N_iteration);
        save('AA_weightMaxtrix4.mat','weightMatrix4','bias4Hidden4','bias4Visible4');

    end
end

end

