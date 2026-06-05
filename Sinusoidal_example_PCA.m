%% Description
% This code implements PCA based model extraction for a
% five-variable dataset. 
% For more details the AEO Paper can be referred: https://arxiv.org/abs/2402.14031

%% Notations
% Number of input variables:                 n 
% Number of neurons in the hidden layer:     h 
% Number of latent variables:                m 
% Number of observations or samples of data: N 
% Number of observations in training data:   Ntr 
% Number of observations in testing data:    Nts 

%%  Generating training and testing dataset
clear all
close all
rand('state',1)
randn('state',1)
N=500;
for j=1:N
    Xraw(j,1:3)=1-2*rand(1,3);
    Xraw(j,4)=sin(3*Xraw(j,1))+0.1*randn(1);
    Xraw(j,5)=Xraw(j,2)+tan(0.5*Xraw(j,3))+0.1*randn(1);
end
Xrawtr=Xraw(1:300,:);    % training data
Xrawts=Xraw(301:500,:);  % testing data

% Normalizing the training data
[Ntr,n]=size(Xrawtr);
Etr=mean(Xrawtr);
Vtr=std(Xrawtr);
Xtr0=Xrawtr-Etr(ones(Ntr,1),:);  % Data is mean-centered
Xtrn=Xtr0./Vtr(ones(Ntr,1),:);   % Data is now normalized
Xtr=Xtrn';                    % This makes observations as column vectors and Xtr is of size n by Ntr

% Normalizing the testing data
[Nts,n]=size(Xrawts);
Ets=mean(Xrawts);
Vts=std(Xrawts);
Xts0=Xrawts-Ets(ones(Nts,1),:);  % Data is mean-centered
Xtsn=Xts0./Vts(ones(Nts,1),:);   % Data is now normalized
Xts=Xtsn';                    % This makes observations as column vectors and Xts is of size n by Nts


%% Model extraction with PCA
p=3;     % Number of independent variables in the input data
corr_mat=cov(Xtr');
[vec,val]=eig(corr_mat);
eigen_val = diag(val);
for i=1:n
    eigenvalues(i)=eigen_val(n-i+1);
    v(:,i)=vec(:,n-i+1);
end

for i=1:n
    contrib(i)=eigenvalues(i)/sum(eigenvalues);
end
Pp= v(:,1:p);
Pr= v(:,p+1:n);
Xhat=Pp*Pp'*Xtr;

% Solving the linear equation       
Xr0=Xtr(p+1:n,:)+0.1*randn(n-p,Ntr);
fun0=@(XrPCA)Pr'*[Xtr(1:p,:);XrPCA];
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e7,'FunctionTolerance',1e-2,'Algorithm','trust-region-dogleg')
[XrPCA,fval0,flag0] = fsolve(fun0,Xr0,options1);
msep= mse(Xtr(p+1:n,:),XrPCA);
mser=mse(Xtr,Xhat);

%% Plotting results
figure(1)
    plot(Xtr(1,:),Xtr(4,:),'r.','LineWidth',.8)
    hold on
    plot(Xtr(1,:),XrPCA(1,:),'b.','LineWidth',.8)
    hold on
    xlabel('$x_{1}$','Interpreter','latex','fontsize',20);ylabel('$x_{4}$','Interpreter','latex','fontsize',20);
    legend('$x_{1}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{4}$','$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_4}(PCA)$', '$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_4}(PCA)$', 'Interpreter','latex','fontsize',15);
    ax = gca;
    ax.GridAlpha = 0.5
    ax.GridLineStyle = ':'
    grid on