% % This files is the final project for 'Certificate in Quantitative Finance
% The code represents the calculation method for Nth-to-Default Basket CDS pricing
%clc

tic %timer start

%enables GPU computing
GPU = gpuDevice(1);


%%The first step is to define the global parameters, such as interest rate and unique recovery rate.

RecoveryRate=0.40;
InterestRate=0.01;
McSim=5000;
m=1;
n=1;
p=1;
%Generates an MonteCarlo Simulation * 5 sized array of random normal variables
    %RandomNormalVariables=gpuArray(randn(McSim,5));

% Random normal variables multiplied by the Cholesky decomposition of the correlation matrix
    %CholeskyXRandomNormalVariables=RandomNormalVariables*Cholesky;   

%Generates correlated uniform variables 
    %CorrelatedUniform=normcdf(CholeskyXRandomNormalVariables); 

CorrelatedUniform=gpuArray(copularnd('Gaussian',Correl,McSim));


Isdefault=gpuArray(zeros(McSim,5));

WhenDefault=gpuArray(zeros(McSim,5));

%Sweeps for default in a McSim x 25 matrix and WhenDefault gives exact
%times
for p=1:5
   for m=1:McSim
    
         if abs(log(1-CorrelatedUniform(m, p))) < DefaultProb(p, 6)           
            Isdefault(m, p * 5) = 5;
            WhenDefault(m,p)=5;            
            for n = 1:5
                     if abs(log(1-CorrelatedUniform(m, p))) < DefaultProb(p, 6 - n)
                         Isdefault(m, p * 5 - n) = 5 - n;
                         WhenDefault(m,p)=5-n;
                     end
             end
         end
    end
end
%finds tau
SimTau=NaN(McSim,10,'gpuArray');
for m = 1:5
    for n = 1:McSim
        if WhenDefault(n, m) > 0
                Log1MinusU = 1 - CorrelatedUniform(n, m);
                a=WhenDefault(n, m);
                SimTau(n, 0 + m) =-1 / HazardRate(m, a)* log(Log1MinusU / SurvivalProbability(m, a));
                SimTau(n, 5 + m) = (WhenDefault(n, m) - 1) + SimTau(n, 0 + m);
        end
    end
end


% gathers exact default time
DefaultTimes=NaN(McSim,5,'gpuArray');
for m=1:5
    DefaultTimes(:,m)= SimTau(:,5+m);
end
DefaultTimes=sort(DefaultTimes,2);



%Default leg  
DefaultLeg=NaN(McSim,5,'gpuArray');
parfor n = 1:5
    for m = 1:McSim
        if DefaultTimes(m, n) >0
            DefaultLeg(m, n) = (1 - RecoveryRate) * exp(-InterestRate * DefaultTimes(m, n)) ; %
        else
            DefaultLeg(m, n) = 0;
        end 
    end
end


%Premium leg
PremiumLeg=zeros(McSim,5,'gpuArray');
parfor n = 1:5
    for m = 1:McSim
        if DefaultTimes(m, n) >0
            PremiumLeg(m, n) = exp(DefaultTimes(m, n)*-InterestRate)*DefaultTimes(m, n);
        else
            PremiumLeg(m, n) = 4.7561;
        end 
    end
end





Basket_CDS_Prices=gpuArray(zeros(1,5));
parfor m=1:5
   Basket_CDS_Prices(1,m)=mean(DefaultLeg(:,m))/mean(PremiumLeg(:,m))* 10000;
end


dummy=gather(Basket_CDS_Prices);
clear Basket_CDS_Prices

Basket_CDS_Prices.First_To_Default=dummy(1);
Basket_CDS_Prices.Second_To_Default=dummy(1,2);
Basket_CDS_Prices.Third_To_Default=dummy(1,3);
Basket_CDS_Prices.Fourth_To_Default=dummy(1,4);
Basket_CDS_Prices.Fifth_To_Default=dummy(1,5);

Basket_CDS_Prices


%GPU's memory cleaned
reset(GPU);                                       




toc %timer stop