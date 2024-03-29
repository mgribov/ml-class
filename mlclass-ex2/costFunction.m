function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i = 1:m
    h(i,1) = sigmoid(theta(1) + theta(2) * X(i,2) + theta(3) * X(i, 3));
end
dif = h - y;
c1 = y' * log(h);
c2 = (1-y)' * log(1 - h);
s = sum( c1 + c2);

J = -1/m * s;

[rows, cols] = size(theta);
for i=1:rows
    s = sum( dif' * X(:,i) );
    grad(i,1) = 1/m * s;
end



% =============================================================

end
