function [model] = trainKNN(X, y, findBest, k, dist, std)
% trainKNN
%   fitkcnn wrapper that returns a model
%   findBest: If true, the input arguments will be ignored and a will
%   automatically find best parameters.
arguments (Input)
    X
    y
    findBest
    k = 5
    dist = 'euclidean'
    std = 'false'
end

arguments (Output)
    model
end

if findBest == false
    fprintf('Training with user values: k=%d, dist=%s, std=%d\n', k, dist, std);
    model = fitcknn(X, y, ...
        'NumNeighbors', k, 'Distance', dist, ...
        'Standardize', std, 'OptimizeHyperparameters', 'none');
    return;
end

fprintf("Running auto fitter\n");
kValues = [1, 3, 5, 7, 9, 15];
distMetrics = {'euclidean', 'cosine', 'hamming'};
standardise = [true, false];

bestLoss = Inf;
bestK = 1;
bestDist = 'euclidean';
bestStandardise = false;

for s = 1:numel(standardise)
    for d = 1:numel(distMetrics)
        for i = 1:numel(kValues)
            model_temp = fitcknn(X, y, ...
                 "NumNeighbors", kValues(i), ...
                 "Distance", distMetrics{d}, ...
                 "Standardize", standardise(s));

            cv = crossval(model_temp, 'KFold', 5);
            loss = kfoldLoss(cv);

            if loss < bestLoss
                bestLoss = loss;
                bestK = kValues(i);
                bestDist = distMetrics{d};
                bestStandardise = standardise(s);
            end
        end
    end
end

fprintf('Best: k=%d, dist=%s, std=%d, CV error=%.4f\n', ...
    bestK, bestDist, bestStandardise, bestLoss);
model = fitcknn(X, y, "NumNeighbors", bestK, "Distance", bestDist, "Standardize", bestStandardise);

end
