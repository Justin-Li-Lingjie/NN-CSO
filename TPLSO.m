classdef TPLSO < ALGORITHM
% <multi/many> <real> <large/none> <constrained/none>
% Competitive swarm optimizer
% phi --- 0.1 --- Social factor

%------------------------------- Reference --------------------------------
% R. Cheng and Y. Jin, A competitive swarm optimizer for large scale
% optimization, IEEE Transactions on Cybernetics, 2014, 45(2): 191-204.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            phi = Algorithm.ParameterSet(0.1);
            
            %% Generate random population
            Population = Problem.Initialization();
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                % Determine the losers and winners
                Fitness = calFitness(Population.objs);
                rank    = randperm(length(Population),floor(length(Population)/3)*3);
                Loser2   = rank(1:end/3);
                Loser1   = rank(end/3+1:end/3*2);
                Winner  = rank(end/3*2+1:end);
                Change = Fitness(Loser1) < Fitness(Loser2);
                Temp   = Loser2(Change);
                Loser2(Change) = Loser1(Change);
                Loser1(Change)= Temp;
                Change = Fitness(Loser1) >= Fitness(Winner);
                Temp   = Winner(Change);
                Winner(Change) = Loser1(Change);
                Loser1(Change)  = Temp;
                Change = Fitness(Loser1)< Fitness(Loser2);
                Temp   = Loser2(Change);
                Loser2(Change) = Loser1(Change);
                Loser1(Change)= Temp;
%                 replace = FitnessSingle(Population(loser)) < FitnessSingle(Population(winner));
%                 temp            = loser(replace);
%                 loser(replace)  = winner(replace);
%                 winner(replace) = temp;
                % Update the losers by learning from the winners
                Loser1Dec  = Population(Loser1).decs;
                Loser2Dec  = Population(Loser2).decs;
                WinnerDec = Population(Winner).decs;
                Loser1Vel  = Population(Loser1).adds(zeros(size(Loser1Dec)));
                Loser2Vel  = Population(Loser2).adds(zeros(size(Loser2Dec)));
                WinnerVel  = Population(Winner).adds(zeros(size(WinnerDec)));
                R1  = repmat(rand(length(Population)/3,1),1,Problem.D);
                R2  = repmat(rand(length(Population)/3,1),1,Problem.D);
                R3  = repmat(rand(length(Population)/3,1),1,Problem.D);
                Loser1Vel = R1.*Loser1Vel + R2.*(WinnerDec-Loser1Dec) + phi.*R3.*(repmat(mean(Population.decs,1),length(Population)/3,1)-Loser1Dec);
                Loser1Dec = Loser1Dec + Loser1Vel;
                Loser2Vel = R1.*Loser2Vel + R2.*(WinnerDec-Loser2Dec) + phi.*R3.*(Loser1Dec-Loser2Dec);
                Loser2Dec = Loser2Dec + Loser2Vel;
                Population(Loser1) = SOLUTION(Loser1Dec,Loser1Vel);
                Population(Loser2) = SOLUTION(Loser2Dec,Loser2Vel);
%                 Population(Winner) = SOLUTION(WinnerDec,WinnerVel);
                Fitness = calFitness(Population.objs);
                [~,index] = sort(Fitness,'descend');
                for j=3:Problem.N
                    r = randperm(j-1,2);
                    r1 = r(1);
                    r2 = r(2);
                    if r1>r2
                        temp = r1;
                        r1 = r2;
                        r2 = temp;
                    end
                    R1  = rand(1,Problem.D);
                    R2  = rand(1,Problem.D);
                    R3  = rand(1,Problem.D);
                    ParticleVelOfj = zeros(1,Problem.D);
                    ParticleVelOfj = R1.* ParticleVelOfj + R2.*(Population(index(r1)).dec-Population(index(j)).dec)+phi.*R3.*(Population(index(r2)).dec-Population(index(j)).dec);
                    ParticlejDec = Population(index(j)).dec + ParticleVelOfj;
                    Population(index(j)) = SOLUTION(ParticlejDec);
                end
            end
        end
    end
end
function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density

    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end