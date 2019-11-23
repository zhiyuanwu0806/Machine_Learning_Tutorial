function PrincipalComponentAnalysis_Reconstruct()
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
% images=loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
% labels=loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');

Number_Sample=length(labels);
inputNodeSize=length(images(:,1));

% % Avergae
% Img_avg=zeros(size(images(:,1)));
% for ii=1:Number_Sample
%     Img_avg=Img_avg+images(:,ii);
% end
% Img_avg=Img_avg/Number_Sample;
% 
% Cov_avg=zeros(inputNodeSize);
% for ii=1:Number_Sample
%     Cov_avg=Cov_avg+(images(:,ii)-Img_avg)*(images(:,ii)-Img_avg)';
% end
% Cov_avg=Cov_avg/Number_Sample;

% [U_space D_eig V_s]=svd(Cov_avg);

load('PCA_result.mat','Img_avg','U_space','D_eig','V_s');

% Select a sample randomly.
% cur_sample=round(rand(1)*10000);
cur_sample=5;
currentLabel=labels(cur_sample)
inputNodes=images(:,cur_sample);

% Show the original sample
xx=reshape(inputNodes,28,28);
figure(1);
image(xx*255);

% Reconstruct
N_row=256; %69;  102; 251; 784/2;
image_h=inputNodes-Img_avg;
curPrj=(U_space(1:N_row,:)*image_h);
rImage=U_space(1:N_row,:)'*curPrj+Img_avg;
%[ix mv]=find(rImage<0.2);  rImage(ix)=0;
% [ix mv]=find(rImage>1);  rImage(ix)=1;
xx=reshape(rImage,28,28);
figure(2);
image(xx*255);

end

