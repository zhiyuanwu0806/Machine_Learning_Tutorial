function RBM_Update()
% This function to train restricted Boltzmann machine (RBM).

% Some parameters
alf=0.01;
elp=0.01;

addpath ../MNIST
% Load Images
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% Load Labes
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

% numberSamples=length(labels);
numberSamples=100;

% Design restricted Boltzmann Machine
inputNodeSize=784;  % Visible nodes
outputLayer=10; % Hidden nodes

% Initialize weight matrices
weightMatrix=alf*(rand(outputLayer,inputNodeSize)-0.5);

% Hidden nodes at stage (1)
h_1=zeros(outputLayer,1);
% Visible node at stage (2)
v_2=zeros(inputNodeSize,1);
% Hidden nodes at stage (2)
h_2=zeros(outputLayer,1);

for it=1:10
    for nn=1:numberSamples
        % Load one sample
        currentLabel=labels(nn)
        % inputNodes=rd_images(:,nn);
        inputNodes=images(:,nn);
        
        % Find h_(1)        for jj=1:outputLayer
            % Calculate P(h_(1)j|v_(1))
            etemp=weightMatrix(jj,:)*inputNodes;
            ptemp=Sigmoid(etemp);
            % Sample h_(1)j
            if rand(1)<ptemp
                h_1(jj)=1;
            else
                h_1(jj)=0;
            end
        end
        % Find v_(2)
        for ii=1:inputNodeSize
            % Calculate P(v_(2)i|h_(1))
            etemp=weightMatrix(:,ii)'*h_1;
            ptemp=Sigmoid(etemp);
            % Sample v_(2)i
            if rand(1)<ptemp
                v_2(ii)=1;
            else
                v_2(ii)=0;
            end
        end
        % Find h_(2)
        for jj=1:outputLayer
            % Calculate P(h_(2)j|v_(2))
            etemp=weightMatrix(jj,:)*v_2;
            ptemp=Sigmoid(etemp);
            % Sample h_(1)j
            if rand(1)<ptemp
                h_2(jj)=1;
            else
                h_2(jj)=0;
            end
        end
        d_weightMatrix=elp*(h_1*inputNodes'-h_2*v_2');
        weightMatrix=weightMatrix+d_weightMatrix;
    end
end
save('rBM_weightMaxtrix.mat','weightMatrix');
end

