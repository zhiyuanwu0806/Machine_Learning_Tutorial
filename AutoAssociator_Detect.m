function AutoAssociator_Detect()
% This function to detect by trained auto-associator.

addpath ../MNIST
% Load Images & Labels
% Training samples
% images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');
% Test samples
images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

% numberSamples=length(labels);
numberSamples=1000;
ix_start=100;
Ix_test=ix_start:ix_start+numberSamples;

% Load trained auto-associator
load('AA_weightMaxtrix1.mat','weightMatrix1','bias4Hidden1','bias4Visible1');
load('AA_weightMaxtrix2.mat','weightMatrix2','bias4Hidden2','bias4Visible2');
load('AA_weightMaxtrix3.mat','weightMatrix3','bias4Hidden3','bias4Visible3');
load('AA_weightMaxtrix4.mat','weightMatrix4','bias4Hidden4','bias4Visible4');
% Whether to show the results
show_result=1;

for nn=Ix_test
	% Load one sample
	currentLabel=labels(nn)
	inputNodes=images(:,nn);
    
%     % For test only
%     if currentLabel~=0
%         continue;
%     end
	
    if show_result==1
        % Show input figure
        xx=reshape(inputNodes,28,28);
        figure(1);
        image(xx*255);
    end

	% Forward propogation
    % Layer 1
	A_hidden=weightMatrix1*inputNodes+bias4Hidden1;
	HiddenNodes1=Sigmoid(A_hidden);
    % Layer 2
    A_hidden=weightMatrix2*HiddenNodes1+bias4Hidden2;
    HiddenNodes2=Sigmoid(A_hidden);
    % Layer 3
    A_hidden=weightMatrix3*HiddenNodes2+bias4Hidden3;
    HiddenNodes3=Sigmoid(A_hidden);
    % Outlayer
    A_hidden=weightMatrix4*HiddenNodes3+bias4Hidden4;
    OutputNodes=Sigmoid(A_hidden);
    
    disp(OutputNodes.');
    
    % Re-construction
    % Outlayer Reconstruction
	A_hidden_hat=weightMatrix4.'*OutputNodes+bias4Visible4;
	rHiddenNodes3=Sigmoid(A_hidden_hat);
    % Layer 3 Reconstruction
	A_hidden_hat=weightMatrix3.'*rHiddenNodes3+bias4Visible3;
	rHiddenNodes2=Sigmoid(A_hidden_hat);
    % Layer 2 Reconstruction
	A_hidden_hat=weightMatrix2.'*rHiddenNodes2+bias4Visible2;
	rHiddenNodes1=Sigmoid(A_hidden_hat);
    %  Layer 1 Reconstruction
	A_hidden_hat=weightMatrix1.'*rHiddenNodes1+bias4Visible1;
	rInputNodes=Sigmoid(A_hidden_hat);

    if show_result==1
        % Show output figure
        xx=reshape(rInputNodes,28,28);
        figure(2);
        image(xx*255);
        reply=input('Do you want to end? Y/N [N]:','s');
        if (reply=='Y')
            return;
        end
    end
		
end
end

