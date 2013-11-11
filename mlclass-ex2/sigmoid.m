function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
val = -z;
[rows, cols] = size(val);
for i=1:rows
    for j=1:cols
        res(i,j) = 1 / (1 + e ^ val(i,j));
    end
end
g = res;

% =============================================================

end
