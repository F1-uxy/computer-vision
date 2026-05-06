function [X, y] = bovw_encode(images, imageSize, vocab, bovwConfig)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    images,
    imageSize,
    vocab,
    bovwConfig
end

arguments (Output)
    X
    y
end

numWords = bovwConfig.numWords;
stepSize = bovwConfig.stepSize;
numImages = numel(images.Files);
X = zeros(numImages, numWords, 'single');
y = images.Labels;

for i = 1:numImages
    img = readimage(images, i);

    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, imageSize);

    [rows, cols] = size(img);
    [gx, gy] = meshgrid(stepSize:stepSize:cols-stepSize, ...
                        stepSize:stepSize:rows-stepSize);
    gridPoints = [gx(:), gy(:)];

    points = SURFPoints(gridPoints);
    [desc, ~] = extractFeatures(img, points, 'Method', 'SURF', 'Upright', true);
    desc = single(desc);

    eucliDistance = pdist2(double(desc), double(vocab));
    [~, wIdx] = min(eucliDistance, [], 2);

    imageHist = histcounts(wIdx, 1:numWords+1);
    total = sum(imageHist);
    imageHist = imageHist / total;
    X(i, :) = single(imageHist);
end

end