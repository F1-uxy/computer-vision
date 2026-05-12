function [X, y] = extractGLCMFeatures(imds, imageSize, useGray)
% extractGLCMFeatures
%   Extracts texture features using multi-direction GLCM statistics

arguments (Input)
    imds
    imageSize
    useGray
end

arguments (Output)
    X
    y
end

numImages = numel(imds.Files);

y = imds.Labels;

distances = [1 2 4];

offsets = [];

for d = distances
    offsets = [offsets;
         0  d;
         d  0;
         0 -d;
        -d  0;
         d  d;
        -d -d;
         d -d;
        -d  d];
end

numOffsets = size(offsets, 1);
numStats = 4; % Contrast, Correlation, Energy, Homogeneity
D = numOffsets * numStats;

X = zeros(numImages, D, 'single');
mu = mean(X);
sigma = std(X);

X = (X - mu) ./ (sigma + eps);

for i = 1:numImages

    if mod(i, 50) == 0 || i == 1
        fprintf('GLCM: processing image %d / %d\n', i, numImages);
    end

    img = readimage(imds, i);

    % ensure grayscale (GLCM works on intensity only)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, imageSize);
    img = uint8(img);
    img = imgaussfilt(img, 0.5);

    feat = [];

    % compute GLCM features across all directions
    for k = 1:numOffsets

        glcm = graycomatrix(img, ...
            'Offset', offsets(k, :), ...
            'NumLevels', 16, ...
            'Symmetric', true);

        stats = graycoprops(glcm, ...
            {'Contrast','Correlation','Energy','Homogeneity'});

        feat = [feat, ...
            stats.Contrast, ...
            stats.Correlation, ...
            stats.Energy, ...
            stats.Homogeneity];
    end

    X(i, :) = single(feat);
end

end