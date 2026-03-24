function [model] = trainKNN(X, y, k, dist)
% trainKNN
%   fitkcnn wrapper that returns a model
arguments (Input)
    X
    y
    k = 5
    dist = 'euclidean'
end

arguments (Output)
    model
end

model = fitcknn(X, y, 'NumNeighbors', k, 'Distance', dist, 'Standardize', false, 'OptimizeHyperparameters', 'none');

end