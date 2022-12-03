function [] = MaxkCut(InputType, InputData, e1, k, max_time, MemLog)
% Usage: MaxkCut(InputType, InputData, e1, max_time, MemLog)
% (a) If you want to generate k partitions for a random graph,
% execute the following:
% MaxkCut('R', [V,degree], e1, k, max_time, MemLog)
% where V is number of nodes and the average degree of each node equals 'degree'
% Example: MaxkCut('R',[100,4],0.05,3,3600,0);
% 
% (b) If you want to generate k partitions for an input graph, first save the
% '.mat' file in the 'Datasets' directory located in the same path as 'MaxkCut.m'
% and then, execute the following:
% MaxkCut('S', filename, e1, k, max_time, MemLog)
% Example: filename = 'G1.mat';
% MaxkCut('S',filename,0.05,3,3600,0);
% 
% Set
% e1 = epsilon, the relative error to generate the solution to SDP (for more
% information about the parameter, please refer to Section 3 in the paper)
% k = the number of partitions
% Optionally, set
% (i) MemLog = 1 if you want to track memory usage
% (ii) max_time = max time to run the algorithm (in seconds)
% 
% The output is stored in the 'output_MkC' directory


%% Set seed value
rng(10);
tic;

%% Check input data
narginchk(4,6);
if nargin == 4
    max_time = 36000;
    MemLog = 0;
end
if nargin == 5
    MemLog = 0;
end


%% Track memory usage
warning off all;
profile clear
if MemLog == 1, profile -memory on; end

%% Set parameters for computing eigenvector
maxit = 300;
tolerance = 0.1;
opts.isreal = 1;
opts.issym = 1;
opts.maxit = maxit;

%% Generate input graph based on input type
if isequal(InputType,'R')
    if ~isnumeric(InputData(1)) && ~isnumeric(InputData(2))
        disp('Enter valid number of nodes (V) and degree');
        return    
    end
    [V,C] = CreateRandomGraph(InputData,k);
elseif isequal(InputType,'S')
    if isfile(['Datasets/',InputData])
        disp('Reading input data');
        [V,C] = CreateInputGraph(InputData,k);
    else
        disp('Enter valid filename');
        return
    end
else
    disp('Enter valid Input type: R or S');
    return
end
E = CreateEdgeSet(V,C);

%% Set input parameters for FW-GS
disp('Setting parameters');
alpha = V;
nedges = length(E);
nsamples = 10; %=k
c = 6;
M = (c*log(2*V+nedges))/e1;
beta = c*trace(C);
Cf = beta*M*alpha^2;
eta = 10^-4;
lambdamaxC = eigs(C,1,'LR');

%% Initialize the algorithm with identity matrix
%Create matrix of random samples
disp('Creating initial random samples');
x = 1:V;
for ii = 1:k*nsamples
    z = randn(V,1);
    if ii == 1
        Z = z;
    else
        Z = [Z z];
    end
end
%Create linear mapping of objective function [<C,X>, A(X)]
disp('Creating linear mapping');
t = 1;
gamma = 2/(t+2);
v = [ones(V+1,1);zeros(nedges,1)];
v(1) = trace(C);

%% First iteration of the algorithm
disp('Starting FW-GS');
%Compute the gradient of obj function
max_viol = max([abs(M.*(v(2:V+1)-1));M.*(-v(V+2:length(v))-(1/(k-1)))]);
penalty = cat(1,exp(M.*(v(2:V+1)-1)-max_viol),exp(M.*(1-v(2:V+1))-max_viol),exp(M.*(-v(V+2:length(v))-(1/(k-1)))-max_viol));
matrix_entry = cat(1,(beta/sum(penalty))*(penalty(1:V)-penalty(V+1:2*V)), -(beta/sum(penalty))*penalty(2*V+1:length(penalty)));
%Generate the update direction using 'eigs'
opts.tol = (1/4)*((eta*gamma*Cf)/(alpha*lambdamaxC));
if opts.tol >= 1
    opts.tol = tolerance;
end    
[u,l] = eigs(-sparse([x,E(:,1)',E(:,2)'],[x,E(:,2)',E(:,1)'],[matrix_entry(1:V);matrix_entry(V+1:V+nedges)/2;matrix_entry(V+1:V+nedges)/2])+C,1,'LR',opts);
%Compute update direction
h = zeros(V+1+nedges,1);
if l >= 0
    h(1) = alpha*u'*C*u;
    h(2:V+1) = alpha*u.^2;
    for ii = 1:nedges
        h(V+1+ii) = alpha*u(E(ii,1))*u(E(ii,2));
    end
end

%Initial constraint violation
%vecMaxPenalty = [];

%% Run the loop until gap is less than epsilon*Tr(C)
disp('#itr|Duality Gap|Max Constr Viol');
while ( ((h-v)'*[1;-matrix_entry] > e1*trace(C) ) && toc <= max_time)
    
    %Display intermediate status
    if mod(t,1000) == 0
        disp([int2str(t),'|',num2str(round((h-v)'*[1;-matrix_entry],2)),'|',num2str(round(max_viol/M,2))]);
    end
    
    %% Generate samples from gradient and update the samples
    for ii = 1:k*nsamples
        x1 = normrnd(0,1);
        w = u*x1;
        if l >= 0
            Z(:,ii) = sqrt(1-gamma)*Z(:,ii) + sqrt(gamma)*w;
        else
            Z(:,ii) = sqrt(1-gamma)*Z(:,ii);
        end
    end
    
    %% Update the linear map 'v'
    v = (1-gamma)*v + gamma*h;
    t = t+1;
    gamma = 2/(t+2);
    
    %% Compute the update direction using 'eigs'
    
    
    %Compute gradient at the new point
    max_viol = max([abs(M.*(v(2:V+1)-1));M.*(-v(V+2:length(v))-(1/(k-1)))]);
    penalty = cat(1,exp(M.*(v(2:V+1)-1)-max_viol),exp(M.*(1-v(2:V+1))-max_viol),exp(M.*(-v(V+2:length(v))-(1/(k-1)))-max_viol));
    matrix_entry = cat(1,(beta/sum(penalty))*(penalty(1:V)-penalty(V+1:2*V)), -(beta/sum(penalty))*penalty(2*V+1:length(penalty)));
    %Compute eigenvector using eigs
    opts.tol = (1/4)*((eta*gamma*Cf)/(alpha*lambdamaxC));
    if opts.tol >= 1
        opts.tol = 0.1;
    end
    %vecMaxPenalty = [vecMaxPenalty; max_viol];
    [u,l] = eigs(-sparse([x,E(:,1)',E(:,2)'],[x,E(:,2)',E(:,1)'],[matrix_entry(1:V);matrix_entry(V+1:V+nedges)/2;matrix_entry(V+1:V+nedges)/2])+C,1,'LR',opts);
    %Compute update direction
    h = zeros(V+1+nedges,1);
    if l >= 0
        h(1) = alpha*u'*C*u;
        h(2:V+1) = alpha*u.^2;
        for ii = 1:nedges
            h(V+1+ii) = alpha*u(E(ii,1))*u(E(ii,2));
        end
    end
    
end

%% Compute feasible solution with best value
%Generate feasible samples
ineq = max(max(-v(V+2:length(v))-(1/(k-1))*ones(nedges,1)),0);
maxeq = max(v(2:V+1))+ineq;
for ii = 1:k*nsamples
    y = randn;
    z = Z(:,ii)+sqrt(ineq)*y*ones(V,1);
    rndm = randn(V,1);
    w = sqrt(ones(V,1)-((v(2:V+1)+ineq)/maxeq)).*rndm+z/sqrt(maxeq);
    Z(:,ii) = w;
end

assignment = zeros(V,nsamples);
%Assign nodes to partition
for ns = 1:nsamples
    for ii = 1:V
        [~, maxindex] = max(Z(ii,k*(ns-1)+1:k*ns));
        assignment(ii,ns) = maxindex;
    end
end

%Generate k-cut value
e = @(k,n)[zeros(k-1,1);1;zeros(n-k,1)];
BestCut = 0;
AvgCut = 0;
for jj = 1:nsamples
    Cut = 0;
    for ii = 1:nedges
        if assignment(E(ii,1),jj) ~= assignment(E(ii,2),jj)
            Cut = Cut + e(E(ii,1),V)'*C*e(E(ii,2),V);
        end
    end
    Cut = -((2*k)/(k-1))*Cut;
    AvgCut = AvgCut+Cut;
    if Cut >= BestCut
        BestCut = Cut;
    end
end
AvgCut = AvgCut/nsamples;
disp(BestCut);

toc
time = toc;

%% Display output
disp('Bound on duality gap (stopping criteria):');
disp((h-v)'*[1;-matrix_entry]);
sp = v(1);
constr_viol_eq = norm(v(2:V+1) -ones(V,1),inf);
constr_viol_ineq = max(-v(V+2:length(v))-1/(k-1)*ones(nedges,1));

disp('Number of iterations:');
disp(t);

disp('Value of SDP relaxation');
disp(sp);

disp('Maximum violation of constraints (equality,inequality):');
disp(constr_viol_eq);
disp(constr_viol_ineq);

disp('Solution of max-k-cut problem:');
disp(BestCut);

%% Write output

MkC.InputParams.nNodes = V;
MkC.InputParams.nEdges = nedges;
MkC.InputParams.k = k;
MkC.InputParams.epsilon = e1;
MkC.InputParams.StopCrit = e1*trace(C);
MkC.InputParams.MaxTime = max_time;
MkC.Output.SDPObjVal = sp;
MkC.Output.MaxInfeasIneq = max(0,constr_viol_ineq);
MkC.Output.MaxInfeasEq = constr_viol_eq;
MkC.Output.BestkCutValue = BestCut;
MkC.Output.AvgkCutValue = AvgCut;
MkC.NIterations = t;
MkC.Time = time;
if MemLog == 1
    p = profile('info');
    memoryUsed = max([p.FunctionTable.PeakMem]);
    memoryUsed = [num2str(memoryUsed/1024),' kB'];
else
    memoryUsed = 'Memory not logged';
end
MkC.MemoryUsed = memoryUsed;
if (h-v)'*[1;-matrix_entry] > e1*trace(C)
    MkC.Status = 'Maximum time reached';
else
    MkC.Status = 'Approx solution found';
end 

if ~exist('output_MkC','dir'), mkdir('output_MkC'); end
if InputType == 'R'
    filename = ['R-',datestr(now,'dd-mm-yy-HH:MM-'),int2str(V),'-',int2str(nedges)];
    MkC.InputParams.Data = ['Random graph with ',int2str(V),' nodes and ',int2str(nedges),' edges'];
    Problem.C = C;
    save(['Datasets/R-',datestr(now,'dd-mm-yy-HH:MM-'),int2str(V),'-',int2str(length(E)),'.mat'],'Problem');
else
    MkC.InputParams.Data = InputData;
    filename = [datestr(now,'dd-mm-yy-HH:MM-'),InputData];
end
save(['output_MkC/',filename],'MkC','-v7.3');

%% Functions used in the file
    function [V,C] = CreateRandomGraph(InputData,k)
        
        %Create random graph
        disp('Creating random graph');
        
        %Initial density of graph
        V = InputData(1);
        Edges = floor(V*InputData(2)); %Density of graph
        %Create random graph
        idx = randi(V,Edges,2);
        %Create edges to make the graph connected
        s = floor(rand(1,V-1).*(1:V-1))+1;
        t = 2:V;
        idx1 = [s',t'];
        %idx2 - matrix of all the edges of random connected graph
        idx2 = [idx;idx1];
        clear s
        clear t
        clear idx1
        clear idx
        
        %Create a symmetric matrix
        idx2 = [idx2; [idx2(:,2),idx2(:,1)]];
        %Delete repeated edges
        idx2 = unique(idx2,'rows');
        %Delete self-loops
        idx2(idx2(:,1)==idx2(:,2),:) = [];

        %Find degree of every node
        [a,b] = hist(idx2(:,1),unique(idx2(:,1)));
        val = [-1*ones(length(idx2),1);a'];
        idx2 = [idx2;[b,b]];
        clear a
        clear b

        %Create a sparse matrx
        C = sparse(idx2(:,1),idx2(:,2), val*((k-1)/(2*k)),V,V);
    end

    function [V,C] = CreateInputGraph(InputData,k)
        %Read GSet data
        disp('Creating graph from GSet');
        p = load(['Datasets/',InputData]);
        C = p.Problem.A;
        V = length(C);
        C = (spdiags(C*ones(V,1),0,V,V)-C)*((k-1)/(2*k));
    end

    function [E] = CreateEdgeSet(V,C)
        %%%%Generate the list of edges E = (i,j), i<j
        E = zeros(nnz(C),2);
        cnt = 1;
        for i = 1:V
            idx = find(C(i,i+1:V));
            idx = idx+i;      %% adjust index
            for j = 1:length(idx)
                E(cnt,:) = [i,idx(j)];
                cnt = cnt+1;
            end
        end
        E = E(any(E,2),:);
    end


end
