clear all;
% PATTERN RECOGNITION EEL709

% Data Preparation
Xtrain = mmread('subject1_fmri_std.train.mtx');
Xtest = mmread('subject1_fmri_std.test.mtx');
% YtrainId = mmread('subject1_wordid.train.mtx');
% YtestId = mmread('subject1_wordid.test.mtx');
YtrainId = xlsread('a2');
YtestId = xlsread('a1');
IdToFeatureMapCentered = mmread('word_feature_centered.mtx');
IdToFeatureMapStd = mmread('word_feature_std.mtx');
YtrainFeature = IdToFeatureMapCentered(YtrainId,:);
YtestFeature = IdToFeatureMapCentered(YtestId(:,1),:);

%CONSTANTS
numFeature = [20,40,60,80,100,120,140,160,180,200];
NFeature = 21764; NTrain = 300; NTest = 60; Dict = 12; NSem = 218;

o = 10;
NFeatureSelection = numFeature(o);

featureTech = 0; algoTech = 3; evalTech=1;


switch featureTech % FEATURE SELECTION

case 1 %Techinique 1 ---- Mean based
k = floor(NFeature/NFeatureSelection); Xtrainf = []; Xtestf = [];
for i = 1:NFeatureSelection
    if(i<NFeatureSelection)
        A = mean(Xtrain(:,(i-1)*k+1:i*k),2);B = mean(Xtest(:,(i-1)*k+1:i*k),2);
        Xtrainf = [Xtrainf A];Xtestf = [Xtestf B];
    else
        A = mean(Xtrain(:,(i-1)*k+1:NFeature),2);B = mean(Xtest(:,(i-1)*k+1:NFeature),2);
        Xtrainf = [Xtrainf A];Xtestf = [Xtestf B];
    end
end

case 2 %Technique 2 --- PCA
[u,s,v] = fsvd(Xtrain,NFeatureSelection);
Xtrainf = Xtrain*v; Xtestf = Xtest*v;
[Xtrainf mu sigma] = featureNormalize(Xtrainf);
Xtestf = (Xtestf - ones(size(Xtestf,1),1)*mu)./(ones(size(Xtestf,1),1)*sigma);

case 3 %Technique 3 --- Discrminatory based - kNN

for l = 1:NFeature
    Tnum = NTest;k = NTrain;
    Data = Xtrain(:,l);
    Class = YtrainId;
    Ranks = zeros(Tnum,1);
    for i = 1 : Tnum
        Tdata = Xtest(i,1);
        T_class = YtestId(i,1);
        dis = pdist2(Tdata, Data);
        [distances, dis_idx] = sort(dis, 'ascend');
        Knn_class = YtrainId(dis_idx(1:k));
        
        [axl,bxl] = unique(Knn_class, 'first');
        
        Knn_class_unique = Knn_class(sort(bxl));
        Ranks(i,1) = find(Knn_class_unique == T_class,1);
    end
    F_ranks_sixty(l) = sum(Ranks);
end 
[ranks_sixty, rank_sixty_idx] = sort(F_ranks_sixty, 'ascend');

Xtrainf = Xtrain(:,rank_sixty_idx(1:NFeatureSelection));
Xtestf = Xtest(:,rank_sixty_idx(1:NFeatureSelection));
      
otherwise %No Feature Selection
Xtrainf = Xtrain; Xtestf = Xtest;
end


switch algoTech % ALGORITHM

case 1 %Gaussian Naive Bayes
nb = NaiveBayes.fit(Xtrainf, YtrainId);
[cpre output] = predict(nb,Xtestf);
[a outPut] = sort(output,2,'descend');

case 2 % Linear LibSVM

SVMstruct = svmtrain(YtrainId, Xtrainf, '-t 0 -b 1');
[predicted_label, precision, prob_estimates] = svmpredict(YtestId(:,1),Xtestf,SVMstruct, '-b 1');

for i =1 : NTest
classes = SVMstruct.Label;
[new_array, new_arr_idx] = sort(prob_estimates(i,:),'descend');
outPut(i,:) = classes(new_arr_idx);
end  
    
case 3 % k Nearest Neighbour

Data = Xtrainf; Class = YtrainId;
%Ranks = zeros(NTest,1);

for i = 1 : NTest
Tdata = Xtestf(i,:);
%T_class = YtestId(i,1);

dis = pdist2(Tdata, Data);
[distances, dis_idx] = sort(dis, 'ascend');
Knn_class = YtrainId(dis_idx(1:NTrain));

[axl,bxl] = unique(Knn_class, 'first');

Knn_class_unique = Knn_class(sort(bxl));
outPut(i,:)=Knn_class_unique';
%Ranks(i,1) = find(Knn_class_unique == T_class,1);
end
%sum(Ranks - 1)/(59*60)
    
case 4 %Sequential Stochastic Gradient Descent Method
A = Xtrainf; Yt = YtrainFeature;
At = [A -A]; wt = zeros(size(At,2),1); 
lambda = 0.0005; num_iters = 1000; wFinal = [];
 for i = 1:size(Yt,2)
    [wF, J_history] = sgd(At, Yt(:,i), wt, lambda, num_iters);
    wFin = wF(1:size(A,2)) - wF(size(A,2)+1:end);
    wFinal = [wFinal  wFin]; i
 end

A = Xtestf*wFinal;
outPut = finalMap(A,IdToFeatureMapCentered);
end

switch evalTech

case 1 %EVALUATION CRITERIA - 1
Yt = YtestId;
error = evaluate1(outPut,Yt);
accuracy = evaluate2(outPut,Yt);

case 2 %EVALUATION CRITERIA - 2
Yt = YtestId;
accuracy = evaluate2(outPut,Yt);
end

r1(o) = error;
r2(o) = accuracy;




