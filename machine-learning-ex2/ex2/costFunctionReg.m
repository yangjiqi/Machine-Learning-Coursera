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

n = length(theta);
sum_t2 = 0;

for j = 2:n
    sum_t2 = sum_t2 + power(theta(j),2);
end

sum_t2 = sum_t2 * lambda / (2 * m);

for i = 1:m
    h = sigmoid(theta' * X(i,:)');
    J = J + (-y(i) * log(h) - (1-y(i))*log(1-h));
    for j = 1:n
        grad(j) = grad(j) + (h - y(i)) * X(i,j);
    end
end

J = J / m + sum_t2;
grad = grad / m;

for j = 2:n
    grad(j) = grad(j) + lambda * theta(j) / m;
end



% =============================================================

end
