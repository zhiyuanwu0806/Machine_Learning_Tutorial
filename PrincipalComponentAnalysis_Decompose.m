function PrincipalComponentAnalysis_Decompose()
%
% Description: Unsupervised Learning: Principal Component Analysis.
%   Using Eigen-decomposition of covariance matrix of random vector.
%   Random vector shall porject on the space.
%

addpath ../MNIST
addpath ../Common
% Load Training Images & Labels
images=loadMNISTImages('../MNIST/train-images.idx3-ubyte');
% load('reducedImages.mat');  images=rd_images; % Reduced size images
labels=loadMNISTLabels('../MNIST/train-labels.idx1-ubyte');

% Load Test Images & Labels
% images=loadMNISTImages('./MNIST/t10k-images.idx3-ubyte');
% labels=loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte');

Number_Sample=length(labels);
% Number_Sample=1000;
inputNodeSize=length(images(:,1));

% Avergae
Img_avg=zeros(size(images(:,1)));
for ii=1:Number_Sample
    Img_avg=Img_avg+images(:,ii);
end
Img_avg=Img_avg/Number_Sample;

Cov_avg=zeros(inputNodeSize);
for ii=1:Number_Sample
    ii
    Cov_avg=Cov_avg+(images(:,ii)-Img_avg)*(images(:,ii)-Img_avg)';
end
Cov_avg=Cov_avg/Number_Sample;

[U_space D_eig V_s]=svd(Cov_avg);

save('PCA_result.mat','Img_avg','U_space','D_eig','V_s');

end

