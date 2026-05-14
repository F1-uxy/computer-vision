function yhat = predictModel(model, X, yte)

yhat = predict(model, X);

confusionchart(yte, yhat);
title('Confusion Matrix');

end