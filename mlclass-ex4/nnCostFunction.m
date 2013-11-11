function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% cost solution is mostly credit Sheeva Azma
% http://www.ml-class.org/course/qna/view?id=3744

% hidden layer
a2 = sigmoid([ones(m, 1) X] * Theta1'); 

% output layer
a3 = sigmoid([ones(m, 1) a2] * Theta2'); 

% this is the boolean vector of the output
% each column is the digit we're looking at (class)
% each row is the answer for specific X which class it falls into
yy = eye(num_labels)(y,:);

% cost
J = (1/m) * sum(sum(-yy .* log(a3) - (1-yy) .* log(1-a3)));

% regularization
[r1, c1] = size(Theta1);
[r2, c2] = size(Theta2);
l = lambda/(2*m);
s1 = sum(sum(Theta1(:,2:c1) .^2 ));
s2 = sum(sum(Theta2(:,2:c2) .^2 ));
reg = l * (s1 + s2);
J = J + reg;



% backprop
for t=1:m

    % forward pass
    a1 = X(t,:)';
    a1 = [1; a1];

    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);


    % error function

    % boolean output array
    tmp = zeros(size(a3));
    tmp(y(t)) = 1;

    d3 = a3 - tmp;
    d2 = Theta2' * d3 .* [1; sigmoidGradient(z2)];
    d2 = d2(2:end);

    % gradient
    Theta1_grad = Theta1_grad + d2 * a1';
    Theta2_grad = Theta2_grad + d3 * a2';

end

% regularization
reg1 = (lambda/m) * Theta1(:,2:end);
reg2 = (lambda/m) * Theta2(:,2:end);

Theta1_grad(:,1) = Theta1_grad(:,1)/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)/m + reg1;

Theta2_grad(:,1) = Theta2_grad(:,1)/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)/m + reg2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
