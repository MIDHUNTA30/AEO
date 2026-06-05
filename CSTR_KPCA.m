%% Description
% This code illustrates a Kernel PCA based model extraction for 
% continuous stirred tank reactor (CSTR) process. 
% For more details the AutoEncoder Paper can be referred: https://arxiv.org/abs/2402.14031

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
global A n m p Ez Vz
N=500;
V=48;Cd=1;k0=7.08*10^(10);E=29900;R=1.99;dH=-30000;
rho=50;Cp=0.75;U=150;Au=150;rhoj=62.3;Cpj=1;Vj=3.85;CA=0.2345;
D=randn(6,N);
 D(1,:)=40+kron(4*randn(1,N/100),ones(1,100));
 D(2,:)=0.5+kron(0.05*randn(1,N/50),ones(1,50));
 D(3,:)=530+0*kron(53*randn(1,N/20),ones(1,20));
 D(4,:)=56.626+kron(5.66*randn(1,N/50),ones(1,50));
 D(5,:)=530+0*kron(53*randn(1,N/50),ones(1,50));
 D(6,:)=10.6137+0*kron(1.06137*randn(1,N/20),ones(1,20));
Dn0=[0.2345*ones(1,N);600*ones(1,N);590*ones(1,N);0.3*ones(1,N)]+0.01*randn(4,N);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e7,'FunctionTolerance',1e-1,'OptimalityTolerance',1e-2','Algorithm','trust-region-dogleg')
fun0=@(Dn)[(D(1,:)./(Au*Dn(4,:))).*(D(2,:)-Dn(1,:))-Cd*k0*Dn(1,:).*exp(-E./(R*Dn(2,:)));
 (D(1,:)./(Au*Dn(4,:))).*(D(3,:)-Dn(2,:))+(-dH/(rho*Cp))*Cd*k0*Dn(1,:).*exp(-E./(R*Dn(2,:)))-(U*Au*(Dn(2,:)-Dn(3,:)))./(Au*Dn(4,:)*rho*Cp);
 (D(4,:)/Vj).*(D(5,:)-Dn(3,:))+(U*Au*(Dn(2,:)-Dn(3,:)))/(Vj*rhoj*Cpj);
 (Dn(1,:).*(k0*Au*Dn(4,:))).*exp(-E./(R*Dn(2,:)))-D(6,:)];
[Dn,fvaln,flagn] = fsolve(fun0,Dn0,options1);

Xraw=[D;Dn]'+1*[0.4 0.005 5.3 0.566 5.3 0.1061 0.00234 6 5.9 0.003].*rand(N,10);
Xrawtr=Xraw(1:300,:);      % Training data
Xrawts=Xraw(301:500,:);    % Testing data

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
p_kpca = 6;
p=6;

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

% Nonlinear model extraction
Xr0_kpca = Xtr(p+1:end,:) + 0.1*randn(n-p,Ntr);
funKPCA = @(Xr) kpca_function(Xtr,Xr,Alpha_r,sigma,K);
opts = optimoptions('fsolve', ...
    'Algorithm','trust-region-dogleg', ...
    'MaxIterations',1e7, ...
    'MaxFunctionEvaluations',1e7, ...
    'FunctionTolerance',1e-2);
[XrKPCA,~,~] = fsolve(funKPCA,Xr0_kpca,opts);

mse_kPCA = mse(Xtr(p+1:end,:),XrKPCA);



%% Plotting results
close all
figure(1)
plot(Xtr(2,:),Xtr(9,:),'r.','LineWidth',1.5)
hold on
plot(Xtr(2,:),XrKPCA(3,:),'b.','LineWidth',0.5)
hold on
ax = gca;
ax.GridAlpha = 0.5
ax.GridLineStyle = ':'
xlabel('$x_{2}$','Interpreter','latex','fontsize',20);ylabel('$x_{9}$','Interpreter','latex','fontsize',20);
legend('$x_{2}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{9}$','$x_{2} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_9}(KPCA)$', 'Interpreter','latex','fontsize',15);
grid on


%%  KPCA function
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
