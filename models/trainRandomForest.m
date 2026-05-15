function bestModel = trainRandomForest(X, y, findBest)

arguments (Input)
    X
    y
    findBest = false
end

fprintf('Training Random Forest...\n');

if findBest == false

    bestModel = fitcensemble( ...
        X, y, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 200, ...
        'Learners', templateTree('MinLeafSize',5));

    return;
end

fprintf('Running hyperparameter search...\n');

leafSizes = [1 5 10 20];
numTrees  = [50 100 200 400];

bestAcc = -inf;
bestModel = [];
bestNumTrees = 0;
bestLeafSize = 0;

for i = 1:length(leafSizes)
    for j = 1:length(numTrees)

        fprintf('Testing: Leaf=%d, Trees=%d\n', leafSizes(i), numTrees(j));

        model = fitcensemble( ...
            X, y, ...
            'Method', 'Bag', ...
            'NumLearningCycles', numTrees(j), ...
            'Learners', templateTree('MinLeafSize', leafSizes(i)));

        cvModel = crossval(model, 'KFold', 5);
        acc = 1 - kfoldLoss(cvModel);

        if acc > bestAcc
            bestAcc = acc;
            bestModel = model;
            bestNumTrees = numTrees(j);
            bestLeafSize = leafSizes(i);
        end

    end
end

fprintf('Best Hyperparameters: numTrees = %d ; leafSize = %d\n', bestNumTrees, bestLeafSize);
fprintf('Best CV Accuracy: %.4f\n', bestAcc);
end