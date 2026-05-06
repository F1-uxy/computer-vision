function [X, y] = bovw_encode(images, imageSize, vocab, bovwCfg)
% bovw_encode
%   Encodes images into Bag-of-Visual-Words histograms

arguments (Input)
    images
    imageSize
    vocab
    bovwCfg
end

arguments (Output)
    X
    y
end

numImages = numel(images.Files);
numWords  = size(vocab, 1);

if isfield(bovwCfg, 'encodeStep')
    stepSize = bovwCfg.encodeStep;
else
    stepSize = 4;
end

X = zeros(numImages, numWords, 'single');
y = images.Labels;

if size(vocab,2) ~= 64
    vocab = vocab';
end

fprintf('Encoding %d images\n', numImages);

for i = 1:numImages
    
    if mod(i, 50) == 0 || i == 1 || i == numImages
        fprintf('  [%d / %d] encoding...\n', i, numImages);
    end

    img = readimage(images, i);

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

    desc = single(desc);

    if size(desc,1) == 64 && size(desc,2) ~= 64
        desc = desc';
    end

    if size(desc,2) ~= 64
        continue;
    end

    wordIdx = knnsearch(vocab, double(desc));

    histVec = histcounts(wordIdx, 1:numWords+1);

    s = sum(histVec);
    if s > 0
        histVec = histVec / s;
    end

    X(i,:) = single(histVec);
end

fprintf('Encoding complete.\n');

end