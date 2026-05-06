function vocab = bovw_buildVocab(images, imageSize, bovwCfg)
% bovw_buildVocab
%   Builds visual vocabulary using k-means clustering on SURF descriptors

arguments (Input)
    images
    imageSize
    bovwCfg
end

arguments (Output)
    vocab
end

numWords = bovwCfg.numWords;

if isfield(bovwCfg, 'vocabStep')
    stepSize = bovwCfg.vocabStep;
else
    stepSize = 8;
end

if isfield(bovwCfg, 'numTrainImgs')
    numTrainImgs = bovwCfg.numTrainImgs;
else
    numTrainImgs = Inf;
end

numImages = numel(images.Files);

if isfinite(numTrainImgs) && numTrainImgs < numImages
    idx = randperm(numImages, numTrainImgs);
else
    idx = 1:numImages;
end

allDesc = [];

fprintf('Building vocab from %d images...\n', numel(idx));

for i = 1:numel(idx)

    if mod(i, 100) == 0 || i == 1 || i == numel(idx)
        fprintf('  [%d / %d] extracting features...\n', i, numel(idx));
    end

    img = readimage(images, idx(i));

    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    img = imresize(img, imageSize);

    [rows, cols] = size(img);
    [gx, gy] = meshgrid(stepSize:stepSize:cols, ...
                        stepSize:stepSize:rows);

    points = SURFPoints([gx(:), gy(:)]);

    [desc, ~] = extractFeatures(img, points, 'Method', 'SURF');

    if isempty(desc)
        continue;
    end

    desc = double(desc);

    if size(desc,1) == 64 && size(desc,2) ~= 64
        desc = desc';
    end

    if size(desc,2) ~= 64
        continue;
    end

    allDesc = [allDesc; desc];
end

assert(~isempty(allDesc), 'No descriptors collected for vocabulary');

fprintf("Starting KMeans\n");

opts = statset('MaxIter', 100, 'Display', 'final');

[~, vocab] = kmeans(allDesc, numWords, ...
    'Options', opts, ...
    'Replicates', 3, ...
    'Distance', 'sqeuclidean');

vocab = single(vocab);

fprintf('Building complete.\n');

end