function [models] = trainSVM(X, y, findBest, kernel, boxConstraint)
% trainSVM
%   Trains 15 one-vs-all binary linear SVMs, one per scene category.
%   findBest: If true, automatically tunes BoxConstraint via cross-validation
%             and ignores the boxConstraint input argument.
%   kernel:   Kernel type e.g. 'linear'
%   boxConstraint: SVM regularisation parameter C (ignored if findBest=true)


arguments (Input)
    X
    y
    findBest    logical
    kernel      = 'linear'
    boxConstraint = 1
end
arguments (Output)
    models
end

if findBest
    boxConstraint = tuneSVM(X, y, kernel);
end

fprintf('Training one-vs-all SVMs (kernel=%s, C=%.4f)...\n', kernel, boxConstraint);

categories = unique(y);
numClasses  = numel(categories);
models = struct('svmModel', cell(numClasses,1), 'className', cell(numClasses,1));

for i = 1:numClasses
    className    = categories(i);
    binaryLabels = (y == className);

    fprintf('  SVM %d/%d: "%s" vs rest\n', i, numClasses, char(className));

    models(i).svmModel  = fitcsvm(X, binaryLabels, ...
                                  'KernelFunction', kernel, ...
                                  'BoxConstraint',  boxConstraint, ...
                                  'Standardize',    true);
    models(i).className = className;
end

fprintf('Done.\n');
end

function bestC = tuneSVM(X, y, kernel)
% tuneSVM
%   Cross-validates over BoxConstraint values and returns the best one.
%   Mirrors the cross-validation loop in partner's trainKNN findBest block.

cValues = [0.001, 0.01, 0.1, 1, 10, 100];
bestLoss = Inf;
bestC    = 1;

fprintf('Tuning BoxConstraint via 5-fold cross-validation...\n');

for ci = 1:numel(cValues)
    c = cValues(ci);
    categories = unique(y);
    foldLosses = zeros(5, 1);
    cv = cvpartition(y, 'KFold', 5);

    for fold = 1:5
        XtrFold  = X(cv.training(fold), :);
        ytrFold  = y(cv.training(fold));
        XvalFold = X(cv.test(fold), :);
        yvalFold = y(cv.test(fold));

        numClasses = numel(categories);
        scores = zeros(sum(cv.test(fold)), numClasses);
        for i = 1:numClasses
            binaryLabels = (ytrFold == categories(i));
            mdlTemp = fitcsvm(XtrFold, binaryLabels, ...
                              'KernelFunction', kernel, ...
                              'BoxConstraint',  c, ...
                              'Standardize',    true);
            [~, s]     = predict(mdlTemp, XvalFold);
            scores(:,i) = s(:,2);
        end

        [~, winIdx] = max(scores, [], 2);
        yhatFold    = categories(winIdx);
        foldLosses(fold) = mean(yhatFold ~= yvalFold);
    end

    meanLoss = mean(foldLosses);
    fprintf('  C=%.4f -> CV error=%.4f\n', c, meanLoss);

    if meanLoss < bestLoss
        bestLoss = meanLoss;
        bestC    = c;
    end
end

fprintf('Best: C=%.4f, CV error=%.4f\n', bestC, bestLoss);
end
