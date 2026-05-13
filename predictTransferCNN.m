function yhat = predictTransferCNN(netStruct, imdsTest)

net = netStruct.net;

inputSize = net.Layers(1).InputSize(1:2);

augTest = augmentedImageDatastore(inputSize, imdsTest);

yhat = classify(net, augTest);

end