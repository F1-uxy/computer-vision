function generateErrorGallery(imdsTest, ytrue, ypred, classes, tag, outDir)

if ~iscategorical(ytrue)
    ytrue = categorical(ytrue);
end
if ~iscategorical(ypred)
    ypred = categorical(ypred);
end

if iscategorical(classes)
    classList = cellstr(classes(:));
elseif isstring(classes)
    classList = cellstr(classes(:));
else
    classList = classes(:);
end

numRows = numel(classList);
numCols = 9;
accuracy = mean(ytrue == ypred);
trueLabels = cellstr(ytrue(:));
predLabels = cellstr(ypred(:));

fig = figure('Visible', 'off');
fig.Position = [100 100 2200 max(1600, 220 * numRows)];
t = tiledlayout(numRows, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, sprintf('%s | Accuracy: %.2f%%', char(tag), 100 * accuracy));

for c = 1:numRows
    cls = classList{c};

    isTrueClass = strcmp(trueLabels, cls);
    isPredClass = strcmp(predLabels, cls);

    tpIdx = find(isTrueClass & isPredClass);
    fnIdx = find(isTrueClass & ~isPredClass);
    fpIdx = find(~isTrueClass & isPredClass);

    for k = 1:numCols
        ax = nexttile;

        if k <= 3
            samplePool = tpIdx;
            sampleType = 'TP';
            samplePos = k;
        elseif k <= 6
            samplePool = fnIdx;
            sampleType = 'FN';
            samplePos = k - 3;
        else
            samplePool = fpIdx;
            sampleType = 'FP';
            samplePos = k - 6;
        end

        if samplePos <= numel(samplePool)
            sampleIdx = samplePool(samplePos);
            imshow(readimage(imdsTest, sampleIdx), 'Parent', ax);
            title(ax, sprintf('%s | True: %s Pred: %s', ...
                sampleType, char(ytrue(sampleIdx)), char(ypred(sampleIdx))), ...
                'FontSize', 8);
        else
            axis(ax, 'off');
        end

        if k == 1
            ylabel(ax, cls, 'FontWeight', 'bold', 'Interpreter', 'none');
        end
    end
end

saveas(fig, fullfile(outDir, [char(tag) '_gallery.png']));
close(fig);

end
