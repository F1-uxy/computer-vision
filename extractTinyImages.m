function [X, y] = extractTinyImages(images, thumbSize, greyscale)
% extractTinyImages
%   Refactors an array of images and features into a supervector
%   thumSize: Target size of image resize (vector [x,y])
%   greyscale: To convert images to greyscale
arguments (Input)
    images
    thumbSize
    greyscale
end

arguments (Output)
    X
    y
end

numImages = numel(images.Files);

if greyscale
    D = thumbSize(1) * thumbSize(2);
else
    D = thumbSize(1) * thumbSize(2) * 3;
end

X = zeros(numImages, D, 'single');
y = images.Labels;

for i = 1:numImages
    img = readimage(images, i);
    
    if(greyscale)
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
    else
        if size(img, 3) == 1
            img = cat(3, img, img, img);
        end
    end

    img = imresize(img, thumbSize);
    img = single(img) / 255.0;
    X(i, :) = img(:)';
end

end