%% Description
% This code implements the Autoencoder with Ordered Variance (AEO) for a
% five-variable dataset. 
% For more details the AEO Paper can be referred: https://arxiv.org/abs/2402.14031

%% Notations
% Number of input variables:                 n 
% Number of neurons in the hidden layer:     h 
% Number of latent variables:                m 
% Number of observations or samples of data: N 
% Number of observations in training data:   Ntr 
% Number of observations in testing data:    Nts 
% Loss Function:                             J= J1+J2+J3 
% Reconstruction Error term :                J1 
% Variance Regularization term :             J2 
% Weight Regulairzation term :               J3

%% Tuning parameters in the algorithm
% Number of neurons in the hidden layer:                h 
% Scaling factor of the variance regularization term:   alpha 
% Scaling factor of the weight regularization term:     beta
% Weighting matrix of the variance regularization term: Q

%%  Generating training and testing dataset
clear all
close all
rand('state',1)
randn('state',5)
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


%% Selecting tuning parameters 
for i=1:5
h=6;           
alpha=0.001;        % Hyperparameter 1
beta=0.05*i;           % Hyperparameter 2
w=10;               % Hyperparameter 3
epsilon=0.01; % Variance threshold
Q=diag([w,w^2,w^3,w^4,w^5]);  
m=n;
A0=rand(h,2*(n+m)); % Initializing A   % A contain the weights (and biases) of encoder and decoder: A=[A1 A2 A3 A4] 
%% Training AEO  
% Defining the loss function for AEO
fun = @(A)trace((Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))))'*(Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))))))+alpha*trace(((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))'*Q*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))+beta*trace(A'*A);
options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
[A,fval,flag]=fminunc(fun,A0,options);   
Ytr= (A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)); 
Vytr=[var(Ytr(1,:));var(Ytr(2,:));var(Ytr(3,:)),;var(Ytr(4,:));var(Ytr(5,:))];
for s=n:-1:1
   if Vytr(s)>epsilon
       p=s;      % Number of significant variables
       break;
   end    
end       
Eytr=mean(Ytr(p+1:m,:)')';    % Computing mean values of residual latent variables for the training data
% In Xhattr computation, we will be replacing the residual latent variables with its mean Eytr
Xhattr=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Ytr(1:p,:)+A(:,n+m+p+1:n+2*m)*Eytr));
Jx(i,1)=trace((Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))))'*(Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))))));
Jy(i,1)=trace(((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))'*Q*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))));
Jtheta(i,1)=trace(A'*A);
beta_all(i,1)=beta;
Vy_all(:,i)=Vytr;
i_all(i,1)=i;
end


%% Plotting results
figure(1)
    plot(beta_all,log(Jx),'r-*','LineWidth',2)
    hold on
    plot(beta_all,log(Jy),'b-*','LineWidth',2)
    hold on
    plot(beta_all,log(Jtheta),'g-*','LineWidth',2)
    hold on
    xlabel('$\beta$','Interpreter','latex','fontsize',15);ylabel('$log(J_{x}),log(J_{y}),log(J_{\theta})$','Interpreter','latex','fontsize',15);
    legend('$log(J_{x})$','$log(J_{y})$', '$log(J_{\theta})$', 'Interpreter','latex','fontsize',15);
    grid on
    ax = gca;
    ax.GridAlpha = 0.3
    ax.GridLineStyle = ':'

figure(2)
    plot(i_all,Vy_all(:,1),'r-*','LineWidth',2)
    hold on
    plot(i_all,Vy_all(:,2),'b-*','LineWidth',2)
    hold on
    plot(i_all,Vy_all(:,3),'g-*','LineWidth',2)
    hold on
    plot(i_all,Vy_all(:,4),'k-*','LineWidth',2)
    hold on
    plot(i_all,Vy_all(:,5),'m-*','LineWidth',2)
    hold on
    xlabel('$j$','Interpreter','latex','fontsize',15);ylabel('$V_{y_j}$','Interpreter','latex','fontsize',15);
    legend('$\beta=0.05$','$\beta=0.1$', '$\beta=0.15$', '$\beta=0.2$', '$\beta=0.25$', 'Interpreter','latex','fontsize',15);
    grid on
    ax = gca;
    ax.GridAlpha = 0.3
    ax.GridLineStyle = ':'  