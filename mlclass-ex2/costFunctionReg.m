function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

for i = 1:m
    h(i,1) = sigmoid(theta(1) + theta(2) * X(i,2) + theta(3) * X(i, 3));
end
dif = h - y;
c1 = y' * log(h);
c2 = (1-y)' * log(1 - h);
s = sum( c1 + c2);

reg = sum(theta(2:length(theta)) .^2);

J = -1/m * s + (lambda/(2*m) * reg);

[rows, cols] = size(theta);
for i=1:rows
    s = sum( dif' * X(:,i) );
    grad(i,1) = 1/m * s;
end

for a=2:rows
    grad(a,1) = grad(a,1) + (lambda/m * theta(a, 1));




% =============================================================

end
