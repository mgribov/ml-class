function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% the reason for the product:
% we multiply each value of a row of X by corresponding value in all_theta, which gives us our very large linear equation:
% y = t1x1 + t2x2 + ... + t400x400 + 1
% when we run it, we get a matrix, with each column representing a label - a digit in this case
% and each row representing the probability of the corresponding row in X being in that column (that digit)
% size(X * all_theta') <--- 5000x10
% then we simply extract the member with highest value in each row, and its index is same as the label (the digit)
[m_theta_val, m_theta_idx] = max(X * all_theta', [], 2);
p = m_theta_idx;


% =========================================================================


end
