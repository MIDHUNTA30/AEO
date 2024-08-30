%% Description
% This code implements the ResNet Autoencoder with Ordered Variance (RAEO) for a
% two-variable dataset. 
% For more details the AutoEncoder Paper can be referred: https://arxiv.org/abs/2402.14031

%% Notations
% Number of input variables:             n 
% Number of neurons in the hidden layer: h 
% Number of neurons in the bottleneck layer (Number of latent variables):   m 
% Number of observations or samples of data: N 
% Loss Function:                            J= J1+J2+J3 
% Reconstruction Error term :               J1 
% Variance Regularization term :            J2 
% Weight Regulairzation term :              J3

%% Tuning parameters in the algorithm
% Number of neurons in the hidden layer:                h 
% Scaling factor of the reconstruction term:            alpha 
% Scaling factor of the variance regularization term:   beta 
% Scaling factor of the weight regularization term:     gamma
% Weighting matrix of the variance regularization term: Q

%%  Generating the input data
clear all
close all
rand('state',1)
N=100;
for j=1:N
    Xraw(j,1)=1-2*rand(1);
    Xraw(j,2)=tanh(3*Xraw(j,1));
end
% Normalizing the input data
[N,n]=size(Xraw);
Ex=mean(Xraw);
Vx=std(Xraw);
X0=Xraw-Ex(ones(N,1),:);  % Data is mean-centred
Xn=X0./Vx(ones(N,1),:);   % Data is now normalized
X=Xn';                    % This makes observations as column vectors and X is of size n by N

%% Selecting tuning parameters 
h=5;           % Tuning parameter 1    
alpha=1;       % Tuning parameter 2 
beta=0.5;      % Tuning parameter 3
gamma=0.1;     % Tuning parameter 4
m=n;
A0=rand(h,2*(n+m)); % Initializing A
Qq=[1 2 3 4 5 6 7 8 9 10]; % Contains the q values used in Q

%% Simulating RAEO for various q values (for explicit model extraction)
for i=1:length(Qq)
    q=Qq(i); 
    Q=diag([1,1*q^2]);     % Tuning parameter 5

    % Defining the loss function for RAEO
    fun = @(A)alpha*trace((X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))))'*(X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))))))+beta*trace((X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))'*Q*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))+gamma*trace(A'*A);
    % Training the autoencoder RAEO
    options = optimoptions('fmincon','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
    F=[];g=[];Feq=[];geq=[]; nonlcon=@nlcon;lb=[];ub=[];
    [A,fval,flag]=fmincon(fun,A0,F,g,Feq,geq,lb,ub,nonlcon,options);   
    Y= (X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))); 
    J1(:,i)=alpha*trace((X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))))'*(X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))))));
    J2(:,i)=beta*trace((X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))'*Q*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))));
    J3(:,i)=gamma*trace(A'*A);
    Vy=[var(Y(1,:));var(Y(2,:))];
    p=1;
    Eyr=mean(Y(p+1:m,:)')';
    fl(:,i)=flag;
    Vyq(:,i)=Vy;
    Xhat=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Y(1:p,:)+A(:,n+m+p+1:n+2*m)*Eyr));

    % Predicting Xr using the explicit relationship
    p=1;        % Number of independent variables in the input data
    Xr_ex=Eyr-A(:,n+p+1:n+m)'*tanh(A(:,1:p)*X(1,:));
    mseXr_ex(:,i)=mse(X(2,:),Xr_ex);
end

%% Simulating RAEO for various q values (for implicit model extraction)
for i=1:10
    q=Qq(i);
    Q=diag([1,1*q^2]);     % Tuning parameter 4

    % Defining the loss function for RAEO
    fun = @(A)alpha*trace((X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))))'*(X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))))))+beta*trace((X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))'*Q*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))+gamma*trace(A'*A);
    % Training the autoencoder RAEO
    options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
    [A,fval,flag]=fminunc(fun,A0,options);   
    Y= (X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))); 
    J1(:,i)=alpha*trace((X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))))))'*(X-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))))));
    J2(:,i)=beta*trace((X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X)))'*Q*(X+(A(:,n+1:n+m)'*tanh(A(:,1:n)*X))));
    J3(:,i)=gamma*trace(A'*A);
    Vy=[var(Y(1,:));var(Y(2,:))];
    p=1;
    Eyr=mean(Y(p+1:m,:)')';
    fl(:,i)=flag;
    Vyq(:,i)=Vy;
    Xhat=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Y(1:p,:)+A(:,n+m+p+1:n+2*m)*Eyr));
    % Solving the nonlinear equation f(Xp,Xr)=0 
    p=1;        % Number of independent variables in the input data
    Xr0=rand(n-p,N);
    options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6,'Algorithm','trust-region-dogleg')
    fun0=@(Xr)Xr+A(:,n+p+1:n+m)'*tanh(A(:,1:p)*X(1:p,:)+A(:,p+1:n)*Xr)-Eyr;
    [Xr,fval0,flag0] = fsolve(fun0,Xr0,options1);
    mseXr(:,i)=mse(X(2,:),Xr);
end

%% Plotting results
figure(1)
    plot(Qq,Vyq(1,:),'r.-','LineWidth',.7)
    hold on
    plot(Qq,Vyq(2,:),'g.-','LineWidth',.7)
    xlabel('$q$','Interpreter','latex');ylabel('$V_{y}$','Interpreter','latex');
    legend('$V_{y_1}$','$V_{y_2}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':'    
    print -dsvg RAEO1
figure(2)
    plot(Qq,J1,'r.-','LineWidth',.7)
    hold on
    plot(Qq,J2,'g.-','LineWidth',.7)
    hold on
    plot(Qq,J3,'b.-','LineWidth',.7)
    xlabel('$q$','Interpreter','latex');ylabel('$J$','Interpreter','latex');
    legend('$J_{1}$','$J_{2}$','$J_{3}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':'    
    print -dsvg RAEO2    
figure(3)
    subplot(2,1,1)
    plot(X(1,:),X(2,:),'ro','LineWidth',.7) 
    hold on    
    plot(X(1,:),Xr_ex(1,:),'go','LineWidth',.7)
    hold on
    plot(X(1,:),Xr(1,:),'bo','LineWidth',.7) 
    xlabel('$x_{1}$','Interpreter','latex');ylabel('$x_{2}$','Interpreter','latex');
    legend('$x_{1}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{2}$','$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{epr_2}$','$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_2}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':' 
    subplot(2,1,2)
    plot(Qq,mseXr_ex,'g.-','LineWidth',.7)
    hold on
    plot(Qq,mseXr,'b.-','LineWidth',.7)
    xlabel('$q$','Interpreter','latex');ylabel('$MSE$','Interpreter','latex');
    legend('$MSE_{epr}$','$MSE_{pr}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':' 
    print -dsvg RAEO3  
figure(4)
    subplot(2,1,1)
    plot(X(1,:),'r-','LineWidth',.7)
    hold on
    plot(Xhat(1,:),'g-','LineWidth',.7)
    xlabel('$k$','Interpreter','latex');ylabel('$x_{1}$','Interpreter','latex');
    legend('$x_{1}$','$\hat{x}_{1}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':'    
    subplot(2,1,2)
    plot(X(2,:),'r-','LineWidth',.7)
    hold on
    plot(Xhat(2,:),'g-','LineWidth',.7)
    xlabel('$k$','Interpreter','latex');ylabel('$x_{2}$','Interpreter','latex');
    legend('$x_{2}$','$\hat{x}_{2}$', 'Interpreter','latex');
    grid on
    ax = gca;
    ax.GridAlpha = 1
    ax.GridLineStyle = ':'    
    print -dsvg RAEO4  

function [c,ceq] =nlcon(A)
   n=2;p=1;
   c=[];
   ceq =[A(:,p+1:n)];
end      
