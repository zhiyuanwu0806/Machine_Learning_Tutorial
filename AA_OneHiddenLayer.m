function [HiddenNodes,updated_weightMatrix,updated_bias4Hidden,...
    updated_bias4Visible]=AA_OneHiddenLayer(inputNodes,...
    weightMatrix,bias4Hidden,bias4Visible,N_iteration)
% This function to update auto-associator for one hidden layer.

% Learning rate
elp=0.05;

for itr_i=1:N_iteration
    % Forward propogation
    A_hidden=weightMatrix*inputNodes+bias4Hidden;
    HiddenNodes=Sigmoid(A_hidden);
    A_hidden_hat=weightMatrix.'*HiddenNodes+bias4Visible;
    OutputNodes=Sigmoid(A_hidden_hat);
    
    % Backward propogation
    dC_a_hat=OutputNodes-inputNodes;
    dC_h_hat=weightMatrix*dC_a_hat;
    dC_a=dC_h_hat.*HiddenNodes.*(1-HiddenNodes);
    d_weightMatrix=HiddenNodes*dC_a_hat.'+dC_a*inputNodes.';
    
    % Updates
    weightMatrix=weightMatrix-elp*d_weightMatrix;
    bias4Hidden=bias4Hidden-elp*dC_a;
    bias4Visible=bias4Visible-elp*dC_a_hat;
end
updated_weightMatrix=weightMatrix;
updated_bias4Hidden=bias4Hidden;
updated_bias4Visible=bias4Visible;

end

