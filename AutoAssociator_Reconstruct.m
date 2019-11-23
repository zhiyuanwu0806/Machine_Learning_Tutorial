function AutoAssociator_Reconstruct()
% This function to reconstruct one sample by trained auto-associator.


% Load trained auto-associator
load('AA_weightMaxtrix1.mat','weightMatrix1','bias4Hidden1','bias4Visible1');
load('AA_weightMaxtrix2.mat','weightMatrix2','bias4Hidden2','bias4Visible2');
load('AA_weightMaxtrix3.mat','weightMatrix3','bias4Hidden3','bias4Visible3');
load('AA_weightMaxtrix4.mat','weightMatrix4','bias4Hidden4','bias4Visible4');

% Whether to show the results
show_result=1;

outputLayer=16;

OutputNodes0=zeros(outputLayer,1);

% for ii=1:outputLayer
    OutputNodes=OutputNodes0;
    % '6'
    % OutputNodes(9)=1;  OutputNodes(16)=1;
    % '7'
    OutputNodes(10)=1;  OutputNodes(12)=1;
    % '0'
    % OutputNodes(1)=1;  OutputNodes(13)=1; OutputNodes(14)=1;
    
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
        figure(1);
        image(xx*255);
%         reply=input('Do you want to end? Y/N [N]:','s');
%         if reply=='Y'
%             return;
%         end
    end
    
% end
end

