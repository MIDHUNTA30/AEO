%% Description
% This code implements Kernel PCA based model extraction for a
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



%%  Kernel PCA-based model extraction 
sigma = 1;
p_kpca = 3;
p=3;
% Constructing Kernel matrix
K = zeros(Ntr,Ntr);
for i = 1:Ntr
    for j = 1:Ntr
        d = Xtr(:,i) - Xtr(:,j);
        K(i,j) = exp(-(d'*d)/(2*sigma^2));
    end
end

% Centering kernel 
oneN = ones(Ntr)/Ntr;
Kc = K - oneN*K - K*oneN + oneN*K*oneN;

% Eigenvalue decomposition
[Vk,Dk] = eig(Kc);
[eigk,idx] = sort(diag(Dk),'descend');
Vk = Vk(:,idx);

for i = 1:Ntr
    Vk(:,i) = Vk(:,i)/sqrt(eigk(i)+1e-8);
end

Alpha_p = Vk(:,1:p_kpca);
Alpha_r = Vk(:,p_kpca+1:end);

% Nonlinear model extraction with KPCA
Xr0_kpca = Xtr(p+1:end,:) + 0.2*randn(n-p,Ntr);
funKPCA = @(Xr) kpca_function(Xtr,Xr,Alpha_r,sigma,K);

opts = optimoptions('fsolve', ...
    'Algorithm','trust-region-dogleg', ...
    'MaxIterations',1e7, ...
    'MaxFunctionEvaluations',1e7, ...
    'FunctionTolerance',1e-2);

[XrKPCA,~,~] = fsolve(funKPCA,Xr0_kpca,opts);

mse_kPCA = mse(Xtr(p+1:end,:),XrKPCA);

%% Plotting results
figure(1)
    plot(Xtr(1,:),Xtr(4,:),'r.','LineWidth',.8)
    hold on
    plot(Xtr(1,:),XrKPCA(1,:),'b.','LineWidth',.8)
    hold on
    xlabel('$x_{1}$','Interpreter','latex','fontsize',20);ylabel('$x_{4}$','Interpreter','latex','fontsize',20);
    legend('$x_{1}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{4}$', '$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_4}(KPCA)$', 'Interpreter','latex','fontsize',15);
    ax = gca;
    ax.GridAlpha = 0.5
    ax.GridLineStyle = ':'
    grid on
    print -dsvg fig5_KPCA

function F = kpca_function(Xtr,Xr,Alpha_r,sigma,Ktrain)

    [n,N] = size(Xtr);
    p = n - size(Xr,1);

    Kx = zeros(N,N);
    for i = 1:N
        xi = [Xtr(1:p,i); Xr(:,i)];
        for j = 1:N
            d = xi - Xtr(:,j);
            Kx(i,j) = exp(-(d'*d)/(2*sigma^2));
        end
    end

    oneN = ones(N)/N;
    Kxc = Kx - oneN*Kx - Kx*oneN + oneN*Kx*oneN;

    F = Alpha_r' * Kxc;
end    