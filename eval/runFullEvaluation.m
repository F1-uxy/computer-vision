function runFullEvaluation(imdsTest, ytrue, ypred, classes, tag, outDir)

if ~iscategorical(ytrue)
    ytrue = categorical(ytrue);
end
if ~iscategorical(ypred)
    ypred = categorical(ypred);
end

if iscategorical(classes)
    preferredOrder = cellstr(classes(:));
elseif isstring(classes)
    preferredOrder = cellstr(classes(:));
else
    preferredOrder = classes(:);
end

trueLabels = cellstr(ytrue(:));
predLabels = cellstr(ypred(:));
presentLabels = unique([trueLabels; predLabels], 'stable');
extraLabels = presentLabels(~ismember(presentLabels, preferredOrder));
order = [preferredOrder; extraLabels];

confMat = zeros(numel(order), numel(order));
for i = 1:numel(trueLabels)
    trueIdx = find(strcmp(order, trueLabels{i}), 1);
    predIdx = find(strcmp(order, predLabels{i}), 1);

    if isempty(trueIdx) || isempty(predIdx)
        continue;
    end

    confMat(trueIdx, predIdx) = confMat(trueIdx, predIdx) + 1;
end

accuracy = sum(diag(confMat)) / sum(confMat(:));
rowTotals = sum(confMat, 2);
classAccuracy = zeros(numel(order), 1);
for i = 1:numel(order)
    if rowTotals(i) > 0
        classAccuracy(i) = confMat(i, i) / rowTotals(i);
    end
end

disp(tag)
disp("Accuracy: " + accuracy)

fig = figure('Visible','off');
fig.Position = [100 100 1700 900];
annotation(fig, 'textbox', [0.06 0.93 0.62 0.05], ...
    'String', char(tag), ...
    'Interpreter', 'none', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', ...
    'FontSize', 12);
annotation(fig, 'textbox', [0.06 0.89 0.62 0.04], ...
    'String', sprintf('Overall Accuracy: %.2f%%', 100 * accuracy), ...
    'Interpreter', 'none', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold', ...
    'FontSize', 11);

axMat = axes('Parent', fig, 'Position', [0.06 0.10 0.62 0.76]);
imagesc(axMat, confMat);
axis(axMat, 'equal');
axis(axMat, 'tight');
colormap(axMat, parula);
colorbar(axMat);

set(axMat, ...
    'XTick', 1:numel(order), ...
    'YTick', 1:numel(order), ...
    'XTickLabel', order, ...
    'YTickLabel', order, ...
    'TickLabelInterpreter', 'none');
xtickangle(axMat, 45);
xlabel(axMat, 'Predicted Class');
ylabel(axMat, 'True Class');

for r = 1:size(confMat, 1)
    for c = 1:size(confMat, 2)
        text(axMat, c, r, num2str(confMat(r, c)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', 'k');
    end
end

ax = axes('Parent', fig, 'Position', [0.73 0.10 0.24 0.76]);
axis(ax, 'off');
text(ax, 0, 1.02, 'Per-Class Accuracy', 'FontWeight', 'bold', ...
    'FontSize', 11, 'VerticalAlignment', 'top', 'Interpreter', 'none');

for i = 1:numel(order)
    yPos = 1 - ((i - 0.5) / numel(order));
    text(ax, 0, yPos, sprintf('%s: %.1f%%', order{i}, 100 * classAccuracy(i)), ...
        'FontSize', 10, 'VerticalAlignment', 'middle', 'Interpreter', 'none');
end

saveas(fig, fullfile(outDir, tag + "_confusion.png"));
close(fig);

generateErrorGallery(imdsTest, ytrue, ypred, order, tag, outDir);

end
