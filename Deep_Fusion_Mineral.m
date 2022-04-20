%function[Accuracy_overall,Sensitivity_overall,Specificity_overall,Precision_overall,FPR_overall,F1score_overall,MCC_overall,Kappa_overall,comp_time]=deepfeaturesvm();
a=gpuDevice(1);
reset(a);
clc;
clear all;
close all;
warning off;

outputFolder=fullfile('C:\Users\pt\Desktop\mineral classification\mineral_images');
rootFolder=fullfile(outputFolder,'minet');
categories={'biotite','bornite','chrysocolla','malachite','muscovite','pyrite','quartz'};
imds=imageDatastore(fullfile(rootFolder,categories),'labelSource','foldernames');

tbl=countEachLabel(imds);

%[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');
biotite=find(imds.Labels == 'biotite',1);
bornite=find(imds.Labels == 'bornite',1);
chrysocolla=find(imds.Labels == 'chrysocolla',1);
malachite=find(imds.Labels == 'malachite',1);
muscovite=find(imds.Labels == 'muscovite',1);
pyrite=find(imds.Labels == 'pyrite',1);
quartz= find (imds.Labels == 'quartz', 1);


%% To show the Sample
figure;
subplot(3,3,1);
imshow(readimage(imds,biotite));
title('biotite');
subplot(3,3,2);
imshow(readimage(imds,bornite));
title('bornite');
subplot(3,3,3);
imshow(readimage(imds,chrysocolla));
title('chrysocolla');
subplot(3,3,4);
imshow(readimage(imds,malachite));
title('malachite');
subplot(3,3,5);
imshow(readimage(imds,muscovite));
title('muscovite');
subplot(3,3,6);
imshow(readimage(imds,pyrite));
title('pyrite');
subplot(3,3,7);
imshow (readimage(imds,quartz));
title ('quartz');

%% Choose the Pre-trained Network

net = vgg19();

%%
net.Layers(1);
net.Layers(end);
% numel(net.Layers(end).classNames)
[trainingSet, testSet]=splitEachLabel(imds,0.8,'randomize');
imageSize= net.Layers(1).InputSize;
augmentedTrainingSet=augmentedImageDatastore(imageSize, trainingSet,'colorPreprocessing','gray2rgb');
augmentedtestSet=augmentedImageDatastore(imageSize, testSet,'colorPreprocessing','gray2rgb');
%% Show the Deep Feature
% w1=net.Layers(2).Weights;
% w1=mat2gray(w1);
% figure;
% montage(w1)
% title('First Convolutional Layer Weights');

%% feature layer

featureLayer1='fc6';
featureLayer2='fc8';

%%
trainingFeatures1=activations(net,augmentedTrainingSet,featureLayer1,...
    'MiniBatchSize',8, 'OutputAs','columns','ExecutionEnvironment','cpu');
trainingFeatures2=activations(net,augmentedTrainingSet,featureLayer2,...
    'MiniBatchSize',8, 'OutputAs','columns','ExecutionEnvironment','cpu');
trainingFeatures=cat(1,trainingFeatures1,trainingFeatures2);
trainingLabels=trainingSet.Labels;
classifier=fitcecoc(trainingFeatures,trainingLabels,'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures1=activations(net,augmentedtestSet,featureLayer1,...
    'MiniBatchSize',8,'OutputAs','columns','ExecutionEnvironment','cpu');
testFeatures2=activations(net,augmentedtestSet,featureLayer2,...
    'MiniBatchSize',8,'OutputAs','columns','ExecutionEnvironment','cpu');
testFeatures=cat(1,testFeatures1,testFeatures2);
predictLabels=predict(classifier,testFeatures,'observationsIn','columns');
testLabels=testSet.Labels;
confMat=confusionmat(testLabels,predictLabels)
%%plot confusion
figure;
plotconfusion(testLabels,predictLabels)
confMat_percentage=bsxfun(@rdivide,confMat,sum(confMat,2))
Accuracy=mean(diag(confMat_percentage))

c_matrix=confMat_percentage;
  [row,col]=size(c_matrix);
            if row~=col
                error('Confusion matrix dimention is wrong')
            end
            n_class=row;
            switch n_class
                case 2
                    TP=c_matrix(1,1);
                    FN=c_matrix(1,2);
                    FP=c_matrix(2,1);
                    TN=c_matrix(2,2);
                    
                otherwise
                    TP=zeros(1,n_class);
                    FN=zeros(1,n_class);
                    FP=zeros(1,n_class);
                    TN=zeros(1,n_class);
                    for i=1:n_class
                        TP(i)=c_matrix(i,i);
                        FN(i)=sum(c_matrix(i,:))-c_matrix(i,i);
                        FP(i)=sum(c_matrix(:,i))-c_matrix(i,i);
                        TN(i)=sum(c_matrix(:))-TP(i)-FP(i)-FN(i);
                    end
                    
            end
            
            %%
            %Calulations
            %1.P-Positive
            %2.N-Negative
            %3.acuuracy
            %4.error
            %5.Sensitivity (Recall or True positive rate)
            %6.Specificity
            %7.Precision
            %8.FPR-False positive rate
            %9.F_score
            %10.MCC-Matthews correlation coefficient
            %11.kappa-Cohen's kappa
            P=TP+FN;
            N=FP+TN;
            switch n_class
                case 2
                    accuracy=(TP+TN)/(P+N);
                    Error=1-accuracy;
                    Result.Accuracy=(accuracy);
                    Result.Error=(Error);
                otherwise
                    accuracy=(TP)./(P+N);
                    Error=(FP)./(P+N);
                    Result.Accuracy=sum(accuracy);
                    Result.Error=sum(Error);
            end
            RefereceResult.AccuracyOfSingle=(TP ./ P)';
            RefereceResult.ErrorOfSingle=1-RefereceResult.AccuracyOfSingle;
            Sensitivity=TP./P;
            Specificity=TN./N;
            Precision=TP./(TP+FP);
            FPR=1-Specificity;
            beta=1;
            F1_score=( (1+(beta^2))*(Sensitivity.*Precision) ) ./ ( (beta^2)*(Precision+Sensitivity) );
            MCC=[( TP.*TN - FP.*FN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) );...
                ( FP.*FN - TP.*TN ) ./ ( ( (TP+FP).*P.*N.*(TN+FN) ).^(0.5) )] ;
            MCC=max(MCC);
            
            %Kappa Calculation BY 2x2 Matrix Shape
            pox=sum(accuracy);
            Px=sum(P);TPx=sum(TP);FPx=sum(FP);TNx=sum(TN);FNx=sum(FN);Nx=sum(N);
            pex=( (Px.*(TPx+FPx))+(Nx.*(FNx+TNx)) ) ./ ( (TPx+TNx+FPx+FNx).^2 );
            kappa_overall=([( pox-pex ) ./ ( 1-pex );( pex-pox ) ./ ( 1-pox )]);
            kappa_overall=max(kappa_overall);
            
            %Kappa Calculation BY n_class x n_class Matrix Shape
            po=accuracy;
            pe=( (P.*(TP+FP))+(N.*(FN+TN)) ) ./ ( (TP+TN+FP+FN).^2 );
            kappa=([( po-pe ) ./ ( 1-pe );( pe-po ) ./ ( 1-po )]);
            kappa=max(kappa);
            
            
            %%
            %Output Struct for individual Classes
            %  RefereceResult.Class=class_ref;
            RefereceResult.AccuracyInTotal=accuracy';
            RefereceResult.ErrorInTotal=Error';
            RefereceResult.Sensitivity=Sensitivity';
            RefereceResult.Specificity=Specificity';
            RefereceResult.Precision=Precision';
            RefereceResult.FalsePositiveRate=FPR';
            RefereceResult.F1_score=F1_score';
            RefereceResult.MatthewsCorrelationCoefficient=MCC';
            RefereceResult.Kappa=kappa';
            RefereceResult.TruePositive=TP';
            RefereceResult.FalsePositive=FP';
            RefereceResult.FalseNegative=FN';
            RefereceResult.TrueNegative=TN';
            
            
            %Output Struct for over all class lists
            
%             Result.Accuracy=sum(accuracy);
%             Result.Sensitivity=mean(Sensitivity);
%             Result.Specificity=mean(Specificity);
%             Result.Precision=mean(Precision);
%             Result.FalsePositiveRate=mean(FPR);
%             Result.F1_score=mean(F1_score);
%             Result.MatthewsCorrelationCoefficient=mean(MCC);
%             Result.Kappa=kappa_overall;

%Change for call asfunction
            Accuracy_overall=sum(accuracy)
           Sensitivity_overall=mean(Sensitivity)
           Specificity_overall=mean(Specificity)
            Precision_overall=mean(Precision)
           FPR_overall=mean(FPR)
         F1score_overall=mean(F1_score)
            MCC_overall=mean(MCC)
            Kappa_overall=kappa_overall



