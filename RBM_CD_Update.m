function RBM_CD_Update()
% This function to train restricted Boltzmann machine (RBM) using
% Contrastive Divergence (CD).

addpath ../MNIST
% Load Images
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% load('reducedImages.mat');  images=rd_images; % Reduced size images
% Load Labels
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

% numberSamples=length(labels);
% ix_1=find(labels==2);
numberSamples=200;

% Design restricted Boltzmann Machine (RBM)
inputNodeSize=784;  % Number of visible nodes
% inputNodeSize=196;  % Number of visible nodes
outputLayer=576; % Number of hidden nodes
k_Gibbs=50;  % Gibbs sampling step
N_iteration=50;
rand('state',0);

% Some parameters
alf=0.1;
elp=0.1/numberSamples;

% Initialize weight matrices
weightMatrix=alf*(rand(outputLayer,inputNodeSize)-0.5);
bias4Hidden=alf*(rand(outputLayer,1)-0.5);
bias4Visible=alf*(rand(inputNodeSize,1)-0.5);

% Hidden nodes at stage (1)
h_1=zeros(outputLayer,1);  % Samples
P_h_1=zeros(outputLayer,1);  % Sample probabilites
% Visible node at stage (2)
v_2=zeros(inputNodeSize,1);  % Samples
% Hidden nodes at stage (2)
% h_2=zeros(outputLayer,1);
Q_h_2=zeros(outputLayer,1); % Sample probabilites

counter=0;
for nn=1:numberSamples
    for it=1:N_iteration
        % Load one sample, namely v_(1)
%         currentLabel=labels(ix_1(nn))
%         inputNodes=images(:,ix_1(nn));
        currentLabel=labels(nn);
        inputNodes=images(:,nn);

        % Initial
        v_1=inputNodes;

        % Just a counter to show current progress
        counter=counter +1
        
        % Find hidden nodes h_(1) from v_(1)
        for kk=1:k_Gibbs
            for jj=1:outputLayer
                % Calculate P(h_(1)j|v_(1))
                etemp=weightMatrix(jj,:)*v_1+bias4Hidden(jj);
                ptemp=Sigmoid(etemp);
                % Sample h_(1)j
                if rand(1)<ptemp
                    h_1(jj)=1;
                else
                    h_1(jj)=0;
                end
                if kk==1
                    P_h_1(jj)=ptemp;
                end
            end
            % Find visible nodes v_(2) from h_(1)
            for ii=1:inputNodeSize
                % Calculate P(v_(2)i|h_(1))
                etemp=h_1'*weightMatrix(:,ii)+bias4Visible(ii);
                ptemp=Sigmoid(etemp);
                % Sample v_(2)i
                if rand(1)<ptemp
                    v_2(ii)=1;
                else
                    v_2(ii)=0;
                end
            end
            v_1=v_2;
        end
        % Find hidden nodes h_(2) from v_(2)
        for jj=1:outputLayer
            % Calculate P(h_(2)j|v_(2))
            etemp=weightMatrix(jj,:)*v_2+bias4Hidden(jj);
            Q_h_2(jj)=Sigmoid(etemp);
        end
        d_weightMatrix=(h_1*inputNodes'-Q_h_2*v_2');
        err=trace(d_weightMatrix*d_weightMatrix')
        weightMatrix=weightMatrix+elp*d_weightMatrix;
        bias4Hidden=bias4Hidden+elp*(h_1-Q_h_2);
        bias4Visible=bias4Visible+elp*(inputNodes-v_2);
    end
end
save('rBM_CD_weightMaxtrix.mat','weightMatrix','bias4Hidden','bias4Visible');
end

