function [X, y] = extractGLCMFeatures(imds, imageSize)

arguments (Input)
    imds
    imageSize
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
        0 d;
        d 0;
        0 -d;
       -d 0;
        d d;
       -d -d;
        d -d;
       -d  d];
end

numOffsets = size(offsets, 1);
numStats = 4;

pyramidLevels = {
    [1 1];
    [2 2];
    [4 4];
};

numRegions = 0;
for p = 1:numel(pyramidLevels)
    numRegions = numRegions + prod(pyramidLevels{p});
end

D = numRegions * numOffsets * numStats;

X = zeros(numImages, D, 'single');

y = imds.Labels;

for i = 1:numImages

    if mod(i, 50) == 0 || i == 1
        fprintf('GLCM-SP: %d / %d\n', i, numImages);
    end

    img = readimage(imds, i);

    if size(img,3) == 3
        img = rgb2gray(img);
    end

    img = imresize(img, imageSize);
    img = imgaussfilt(uint8(img), 0.5);

    feat = [];

    for p = 1:numel(pyramidLevels)

        grid = pyramidLevels{p};
        rows = grid(1);
        cols = grid(2);

        hStep = floor(size(img,1) / rows);
        wStep = floor(size(img,2) / cols);

        for r = 1:rows
            for c = 1:cols

                r1 = (r-1)*hStep + 1;
                r2 = r*hStep;

                c1 = (c-1)*wStep + 1;
                c2 = c*wStep;

                patch = img(r1:r2, c1:c2);

                for k = 1:numOffsets

                    glcm = graycomatrix(patch, ...
                        'Offset', offsets(k,:), ...
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

            end
        end
    end

    X(i,:) = single(feat);
end

end