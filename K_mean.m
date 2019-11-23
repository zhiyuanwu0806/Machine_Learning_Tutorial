function K_mean()
%
% Description: Unsupervised Learning: K-mean.
%

% rand('state',0);
% randn('state',0);

K_num=3;
mu_x=[1 1; 1 3; 3 2];
var_x=[0.1 0.5; 0.5 0.5; 0.5 2];
L_grp=1000;
% Create K data clusters
grp1=[mu_x(1,1)+randn(L_grp,1)*sqrt(var_x(1,1)) ...
    mu_x(1,2)+randn(L_grp,1)*sqrt(var_x(1,2))];
grp2=[mu_x(2,1)+randn(L_grp,1)*sqrt(var_x(2,1)) ...
    mu_x(2,2)+randn(L_grp,1)*sqrt(var_x(2,2))];
grp3=[mu_x(3,1)+randn(L_grp,1)*sqrt(var_x(3,1)) ...
    mu_x(3,2)+randn(L_grp,1)*sqrt(var_x(3,2))];

group=[grp1;grp2;grp3];

% Show figure
show_figure=1;
if show_figure==1
    figure;
    plot(grp1(:,1),grp1(:,2),'gx'); hold on;
    plot(grp2(:,1),grp2(:,2),'b+'); hold on;
    plot(grp3(:,1),grp3(:,2),'y*'); hold on;
    plot(mu_x(:,1),mu_x(:,2),'rv');
end

% Get random interleaver
[Ix_il,Ix_deil]=srand(L_grp*K_num,7);

% Shuffle all data randomly.
group_int=group(Ix_il,:);

% group space
grp1=group_int(:,1);
grp2=group_int(:,2);
min1=min(grp1); max1=max(grp1); Step1=(max1-min1)/(K_num-1);
min2=min(grp2); max2=max(grp2); Step2=(max2-min2)/(K_num-1);

% Initialization
cent_x=zeros(K_num,2);  % Centers
for ii=1:K_num
    cent_x(ii,:)=[min1+(ii-1)*Step1 min2+(ii-1)*Step2];
end
int_cent=cent_x;
group_ix=zeros(L_grp*K_num,1);  % For group index
dis_total=0;  % total distance
last_dis_total=0; 

for itr=1:50
    itr
    for ll=1:L_grp*K_num
        x_sample=group_int(ll,:);
        % Find which cluster
        min_dis=(x_sample-cent_x(1,:))*(x_sample-cent_x(1,:))';
        group_ix(ll)=1;
        for kk=2:K_num
            dis_temp=(x_sample-cent_x(kk,:))*(x_sample-cent_x(kk,:))';
            if dis_temp<min_dis
                group_ix(ll)=kk;
                min_dis=dis_temp;
            end
        end
        dis_total=dis_total+min_dis;
    end
    % Check convergence
    if abs(dis_total-last_dis_total)<0.1
        break;
    else
        last_dis_total=dis_total;
    end
    % Calculate new centers
    for kk=1:K_num
        ix_temp=find(group_ix==kk);
        cent_x(kk,:)=mean(group_int(ix_temp,:));
    end
end

if show_figure==1
    figure;
    plot(group_int(:,1),group_int(:,2),'k.'); hold on;
    plot(int_cent(:,1),int_cent(:,2),'-m^');
    plot(cent_x(:,1),cent_x(:,2),'rs');
end

mu_x
cent_x

end

