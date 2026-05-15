function [X, y] = extractHOG(images, imageSize, cellSize)
% extractHOG
%   Extracts HOG feature vectors from an image datastore
%   imageSize: Target size to resize images to e.g. [256 256]
%   cellSize:  HOG cell size in pixels e.g. [8 8]
arguments (Input)
    images
    imageSize
    cellSize
end
arguments (Output)
    X
    y
end

numImages = numel(images.Files);

dummy = zeros(imageSize, 'uint8');
dummyFeat = extractHOGFeatures(dummy, 'CellSize', cellSize);
D = numel(dummyFeat);

X = zeros(numImages, D, 'single');
y = images.Labels;

for i = 1:numImages
    img = readimage(images, i);

    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    img = imresize(img, imageSize);

    X(i, :) = single(extractHOGFeatures(img, 'CellSize', cellSize));
end

end
