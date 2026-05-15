function yhat = predictSVM(models, X, yte)
% predictSVM
%   Predicts class labels using one-vs-all SVM models from trainSVM.
%   Each SVM scores a test image (W*X + B), the highest score wins.
arguments (Input)
    models
    X
    yte
end
arguments (Output)
    yhat
end
 
numClasses  = numel(models);
numSamples  = size(X, 1);
scores      = zeros(numSamples, numClasses);
 
for i = 1:numClasses
    [~, s]     = predict(models(i).svmModel, X);
    scores(:,i) = s(:,2); 
end
 
[~, winIdx] = max(scores, [], 2);
classNames  = arrayfun(@(m) char(m.className), models, 'UniformOutput', false);
yhat        = categorical(classNames(winIdx)');

confusionchart(yte, yhat);
title('SVM Confusion Matrix');
 
end