function vocab = bovw_buildVocab(images, imageSize, bovwConfig)
% bovw_buildVocab
%   
arguments (Input)
    images
    imageSize
    bovwConfig
end

arguments (Output)
    vocab
end

numWords = bovwConfig.numWords;
stepSize = bovwConfig.stepSize;
numImages = numel(images.Files);
allDescriptors = [];

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
    [desc, ~] = extractFeatures(img, points, 'Method','SURF', 'Upright', true);

    if ~isempty(desc)
        allDescriptors = [allDescriptors; single(desc)];
    end
end

options = statset('MaxIter', 100, 'Display', 'final');
vocab = single(kmeans(double(allDescriptors), numWords, ...
    'Replicates', 3, ...
    'Options',    options));

end