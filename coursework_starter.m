%% ============INTRO=====================
% INTRO and TASK 1 sections can be run as in the current form as Task 1 functions to be implemented are provided as obscured p-code - see TASK 1.

clear; clc;

baseDir = fileparts(mfilename("fullpath"));
addpath(genpath(baseDir));
C = config();
copyfile(fullfile(C.baseDir, 'config.m'), fullfile(C.outDir, 'configrecord.m'));

trainDir = fullfile(C.dataRoot, "train");
testDir = fullfile(C.dataRoot, "test");

if ~isfolder(trainDir) || ~isfolder(testDir)
    error("Expected dataset folders at %s and %s.", trainDir, testDir);
end

imdsTrain = createLabeledImageDatastore(trainDir);
imdsTest = createLabeledImageDatastore(testDir);

classes = C.classOrder;

%% ================= TASK 1 =================
% You need to write extractTinyImages which is provided as obscured p-code so that you can run the code and see an example results including confusion matrix and image gallery generated.
% You should consider passing more relevant parameters to this function, e.g. 'rgb' or 'grayscale', type of normalisation, type of cropping etc.
%
% Also you need to write trainKNN function which is also provided as obscured p-code
% As above you should consider passing more relevant parameters to this function, e.g. 'k' and 'distance metric'. You should define them in your config file.

% trainKNN
% You can use fitcknn here. You should not use OptimizeHyperparameters option which needs to be disabled 
% i.e. you need to perform fine-tuning of your k distance metric and standardisation parameters manually i.e. 
% you should not enable OptimizeHyperparameters (keep it disabled as it is by default).
% Your fine-tuning should be done using cross-validation on a training set part of the dataset.
% Of course, you need to test your fine-tune model on the test set part of the dataset using the Matlab built in predict function.


[Xtr1, ytr] = extractTinyImages(imdsTrain, C.thumbnailSize, C.greyscale);
[Xte1, yte] = extractTinyImages(imdsTest,  C.thumbnailSize, C.greyscale);

mdl1Path = fullfile(C.modelCacheDir, 'Task1_kNN_model.mat');
if exist(mdl1Path, 'file')
    modelData = load(mdl1Path, 'mdl1');
    mdl1 = modelData.mdl1;
else
    mdl1 = trainKNN(Xtr1, ytr, C.knn.auto, C.knn.k, C.knn.dist, C.knn.std); 
    % save(mdl1Path, 'mdl1'); Enable this one happy with model otherwise
    % you have to delete the cache manually
end

yhat1 = predict(mdl1, Xte1);

runFullEvaluation(imdsTest, yte, yhat1, classes, "Task1_kNN", C.outDir);

%% === TASK 1 TESTERS ===
% Just displays a few tiny'ified images for a sanity check
figure;
for i = 1:9
    img = reshape(Xtr1(i,:), C.thumbnailSize(1), C.thumbnailSize(2), 1); % 1 for greyscale ; 3 for rgb
    subplot(3,3,i);
    imshow(img);
    title(string(ytr(i)));
end

%% ================= TASK 2 =================
% As in Task 1, you need to implement exctractHOG and trainSVM functions.
% As above you should include more parameters. You should define them in config.

mdl2Path = fullfile(C.modelCacheDir, 'Task2_HOG_SVM_model.mat');
if exist(mdl2Path, 'file')
    load(mdl2Path,'Xtr2','Xte2','ytr','yte');
else
    [Xtr2, ytr] = extractHOG(imdsTrain, C.imageSize, C.hog.cellSize);
    [Xte2, yte] = extractHOG(imdsTest,  C.imageSize, C.hog.cellSize);
    save(mdl2Path, 'Xtr2','Xte2','ytr','yte');
end

% trainSVM
% You can use fitcsvm here designed for binary classification. 
% You should train a linear linear SVM for every category (i.e. one vs all)
% and then use the learned linear classifiers to predict the category of
% every test image. Every test feature will be evaluated with all 15 SVMs
% and the most confident SVM will "win". Confidence, or distance from the
% margin, is W*X + B where '*' is the inner product or dot product and W and
% B are the learned hyperplane parameters. 
% You should not use OptimizeHyperparameters option which needs to remain disabled 
% i.e. you need to perform optimisation of your BoxConstraint manually.

% Furthermore, you may notice that this can be also performed in a more automatic way 
% using fitecoc which was designed for multiple class problems. 
% However, you should not use this one or any similar function. 
% That is please write your code and perform fine-tuning for binary one-vs-all linear SVM 
% using fitcsvm with OptimizeHyperparameters switched off.
% The fine-tuning should be done using cross-validation on the training set part of the dataset.
% Of course, you need to test your fine-tune model on the test set part of the dataset.

% You shoud use this approach in all places in your coursework where you use SVM.

mdl2 = trainSVM(Xtr2, ytr, C.svm.kernel);
yhat2 = predict(mdl2, Xte2);

runFullEvaluation(imdsTest, yte, yhat2, classes, "Task2_HOG_SVM", C.outDir);

%% ================= TASK 3 =================
% As in previous tasks you need to implement bovw_buildVocab and bovw_encode functions. You can use trainSVM developed for Task 2.
% As above you should include more parameters that you will need to define in config. 
% Remember you need to perform BoxConstraint fine tuning for SVM.

% function bovw_buildVocab
% 'vocab' should be C.bovw.numWords x 64 (for SURF features). Each row is a cluster centroid / visual word.
% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all training images, although it would be better
% to do so. You can simply create a coarse with a large step size here, but a smaller step size in bovw_encode. 
% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

% function bovw_encode
% This function assumes that vocab exists and is an N x 64
% matrix where each row is a kmeans centroid or a visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.
% Xtr3 and Xte3 are an both 1500 x C.bovw.numWords matrix and 
% they can be also saved to disk to save subsequent recomputations.
% You will want to construct SURF features here in the same way you
% did in bovw_buildVocab (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SURF features will look very different from a smaller version of the same
% image.


mdl3Path = fullfile(C.modelCacheDir, 'Task3_bovw_vocab.mat');
if exist(mdl3Path, 'file')
    load(mdl3Path, 'vocab');
else
    vocab = bovw_buildVocab(imdsTrain, C.imageSize, C.bovw);
    save(mdl3Path, 'vocab');
end

mdl31Path = fullfile(C.modelCacheDir, 'Task3_bovw_features.mat');
if exist(mdl31Path, 'file')
    load(mdl31Path, 'Xtr3','Xte3','ytr','yte');
    extrData3 = data3.extrData3;
else
    [Xtr3,ytr] = bovw_encode(imdsTrain, C.imageSize, vocab, C.bovw);
    [Xte3,yte] = bovw_encode(imdsTest,  C.imageSize, vocab, C.bovw);
    save(mdl31Path, 'Xtr3','Xte3','ytr','yte');
end

mdl3 = trainSVM(Xtr3, ytr, C.svm.kernel);
yhat3 = predict(mdl3, Xte3);

runFullEvaluation(imdsTest, yte, yhat3, classes, "Task3_BoVW_SVM", C.outDir);

%% ================= TASK 4 =================
% As in previous tasks you need to implement trainTranferCNN and predictTransferCNN. 
% You need to perform experiments demonstrating fine-tuning of the pretrained resnet18 network.

netStruct = trainTransferCNN(imdsTrain, classes, C);
yhat4 = predictTransferCNN(netStruct, imdsTest);
yte = imdsTest.Labels;
runFullEvaluation(imdsTest, yte, yhat4, classes, "Task4_TransferCNN", C.outDir);
%% ================= TASK 5 =================
% This one is up to you as described in coursework brief.

%% HELPER FUNCTIONS - no need to edit

function imds = createLabeledImageDatastore(dataDir)
    allowedExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};
    files = {};
    labels = {};
    [files, labels] = collectImageFiles(dataDir, allowedExt, files, labels);

    if isempty(files)
        error("No valid image files were found in %s.", dataDir);
    end

    files = cellfun(@char, files, 'UniformOutput', false);
    labels = cellfun(@char, labels, 'UniformOutput', false);

    imds = imageDatastore(files);
    imds.Labels = categorical(labels);
end

function [files, labels] = collectImageFiles(currentDir, allowedExt, files, labels)
    listing = dir(currentDir);

    for i = 1:numel(listing)
        name = listing(i).name;

        if strcmp(name, '.') || strcmp(name, '..') || (~isempty(name) && name(1) == '.')
            continue;
        end

        fullPath = fullfile(currentDir, name);
        if listing(i).isdir
            [files, labels] = collectImageFiles(fullPath, allowedExt, files, labels);
            continue;
        end

        [parentDir, ~, ext] = fileparts(fullPath);
        if ~any(strcmpi(ext, allowedExt))
            continue;
        end

        [~, labelName] = fileparts(parentDir);
        files{end+1,1} = char(fullPath);
        labels{end+1,1} = lower(char(labelName));
    end
end
