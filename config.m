function C = config()

baseDir = fileparts(mfilename("fullpath"));
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

C.dataRoot = fullfile(baseDir, "data");
C.baseDir = fullfile(baseDir);
C.classOrder = {'bedroom', 'livingroom', 'kitchen', 'store', 'house', ...
    'industrial', 'tallbuilding', 'stadium', 'highway', 'street', ...
    'mountain', 'coast', 'field', 'forest', 'underwater'};

C.useProvidedSplit = false;

C.imageSize = [256 256];

C.thumbnailSize = [32 32];
C.knn.k = 9;

C.hog.cellSize = [8 8];
C.svm.kernel = "linear";

C.bovw.numWords = 200;
C.bovw.stepSize = 8;

%some sample CNN parameters, you will need some extra parameters describing how you do the fine tuning e.g. layers to be frozen
C.cnn.base = "resnet18";
C.cnn.epochs = 5;
C.cnn.miniBatchSize = 16;
C.cnn.initialLearnRate = 1e-4;
C.cnn.l2 = 1e-4;

C.resultsRoot = fullfile(baseDir, "results");
if ~isfolder(C.resultsRoot), mkdir(C.resultsRoot); end

C.modelCacheDir = fullfile(C.resultsRoot, "model_cache");
if ~isfolder(C.modelCacheDir), mkdir(C.modelCacheDir); end

C.outDir = fullfile(C.resultsRoot, timestamp);
if ~isfolder(C.outDir), mkdir(C.outDir); end

end
