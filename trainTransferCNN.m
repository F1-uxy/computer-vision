function netStruct = trainTransferCNN(imdsTrain, classes, C)

% Load pretrained network
net = resnet18;

% Inspect architecture if needed
%analyzeNetwork(net)

inputSize = net.Layers(1).InputSize(1:2);

% Split training set into train/validation

[imdsTrainSplit, imdsVal] = splitEachLabel( ...
    imdsTrain, ...
    0.8, ...
    "randomized");

% Data augmentation

augmenter = imageDataAugmenter( ...
    "RandXReflection", true, ...
    "RandRotation", [-10 10], ...
    "RandXTranslation", [-5 5], ...
    "RandYTranslation", [-5 5]);

augTrain = augmentedImageDatastore( ...
    inputSize, ...
    imdsTrainSplit, ...
    "DataAugmentation", augmenter);

augVal = augmentedImageDatastore(inputSize, imdsVal);

% Convert to layer graph

lgraph = layerGraph(net);

% Replace fully connected layer

newFC = fullyConnectedLayer( ...
    numel(classes), ...
    "Name", "new_fc", ...
    "WeightLearnRateFactor", 10, ...
    "BiasLearnRateFactor", 10);

% Replace classification layer

newClassLayer = classificationLayer("Name","new_classoutput");

lgraph = replaceLayer(lgraph, "fc1000", newFC);
lgraph = replaceLayer(lgraph, "ClassificationLayer_predictions", newClassLayer);


% EXPERIMENT TYPES
%   Choose:
%   "frozen"
%   "partial"
%   "full"

mode = C.transfer.mode;

switch mode

    case "frozen"

        layerNames = string({lgraph.Layers.Name});

        for i = 1:numel(lgraph.Layers)

            layer = lgraph.Layers(i);

            % Skip the new classifier head
            if layer.Name == "new_fc"
                continue;
            end

            % Freeze learnable layers
            if isprop(layer, 'WeightLearnRateFactor')
                layer.WeightLearnRateFactor = 0;
            end

            if isprop(layer, 'BiasLearnRateFactor')
                layer.BiasLearnRateFactor = 0;
            end

            lgraph = replaceLayer(lgraph, layer.Name, layer);
        end

    case "partial"

        for i = 1:numel(lgraph.Layers)

            layer = lgraph.Layers(i);

            % Freeze layers BEFORE res5
            if ~contains(layer.Name, "res5") && ...
               ~contains(layer.Name, "new_fc")

                if isprop(layer, 'WeightLearnRateFactor')
                    layer.WeightLearnRateFactor = 0;
                end

                if isprop(layer, 'BiasLearnRateFactor')
                    layer.BiasLearnRateFactor = 0;
                end

                lgraph = replaceLayer(lgraph, layer.Name, layer);
            end
        end

    case "full"

        % No freezing
end


% Training options

options = trainingOptions("adam", ...
    "MiniBatchSize", C.cnn.miniBatchSize, ...
    "MaxEpochs", C.cnn.epochs, ...
    "InitialLearnRate", C.cnn.initialLearnRate, ...
    "L2Regularization", C.cnn.l2, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augVal, ...
    "ValidationFrequency", 20, ...
    "Verbose", false, ...
    "Plots", "training-progress");

% Train

trainedNet = trainNetwork(augTrain, lgraph, options);

% Store network

netStruct.net = trainedNet;
netStruct.mode = mode;

end