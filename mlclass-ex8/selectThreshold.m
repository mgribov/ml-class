function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions





    % precision = true_pos / (true_pos + false_pos)
    % recall = true_pos / (true_pos + false_neg)
    % f1 = 2 * ( (precision * recall) / (precision + recall) )
    
    % lets see what we think is currently the anomalies based on our epsilon for the cv set
    predictions = (pval < epsilon);

    % compute f1-score components
    false_pos = sum( (predictions == 1) & (yval == 0));
    true_pos = sum( (predictions == 1) & (yval == 1));
    false_neg = sum( (predictions == 0) & (yval == 1));

    precision = true_pos / (true_pos + false_pos);
    recall = true_pos / (true_pos + false_neg);

    F1 = 2 * ( (precision * recall) / (precision + recall) );



    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end