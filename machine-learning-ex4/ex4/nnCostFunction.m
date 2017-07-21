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
X_addBias = [ones(size(X,1),1), X];
h1 =sigmoid(X_addBias * Theta1'); % 5000 x 25
h1_addBias = [ones(size(h1,1),1), h1]; % 5000 x 26
h2 = sigmoid (h1_addBias * Theta2'); % 5000 x 10

% Method 1
% make Y a big 5000 x 10 matrix
y_row = 1:num_labels;
Y = repmat(y_row,m,1);  % m = 5000    num_labels = 10
Y = (Y==y); % 5000 x 10

J = sum(sum(-Y .* log(h2) - (1-Y) .* log(1-h2)))/m;


% % Method 2
% % calculate each y as a 10 x 1 vector
% % Y_eye = eye(num_labels);
% for i = 1: m
%     for j = 1:num_labels
% %         J = J + (-Y_eye(j,y(i)) * log(h2(i,j)) - ...
% %             (1 - Y_eye(j,y(i))) * log(1 - h2(i,j)));
%         J = J + ( - (y(i)==j) * log(h2(i,j)) - ...
%             (1 - (y(i)==j)) * log(1 - h2(i,j)));
%     end
% end
% J = J/m;
% 
% % Method 3
% % only 1 for loop (m)
% Y_eye = eye(num_labels);
% for i = 1:m
%     J = J + (- Y_eye(:,y(i)) .* log((h2(i,:))') - ...
%         (1 - Y_eye(:,y(i))) .* log(1 - (h2(i,:))'));
% end
% J = sum(J)/m;
% for i = 1:m
%     J = J + sum(- Y_eye(:,y(i)) .* log((h2(i,:))') - ...
%         (1 - Y_eye(:,y(i))) .* log(1 - (h2(i,:))'));
% end
% J = J/m;

% Method 4
% only 1 for loop (K)
% use Y for Method 1

% 
% % Solution 1
% t1_noBias = Theta1(:,2:end);
% t2_noBias = Theta2(:,2:end);
% J = J + (sum(sum(power(t1_noBias,2))) + sum(sum(power(t2_noBias,2)))) * ...
%     lambda / (2 * m);


% % Solution 2
% t1_noBias = Theta1(:,2:end);
% eye1 = eye(size(t1_noBias,2)); % must be consistent with the number of columns 
% t2_noBias = Theta2(:,2:end);
% eye2 = eye(size(t2_noBias,2)); % cannot be the number of rows
% J = J + (sum(sum((t1_noBias' * t1_noBias) .* eye1)) + sum(sum((t2_noBias' * t2_noBias) .* eye2))) * ...
%     lambda / (2 * m);

% Solution 3 % Wrong!!!
% t1_noBias = Theta1(:,2:end);
% eye1 = eye(size(t1_noBias,1));
% t2_noBias = Theta2(:,2:end);
% eye2 = eye(size(t2_noBias,1));
% J = J + (sum(sum((t1_noBias * t1_noBias') .* eye1)) + sum(sum((t2_noBias * t2_noBias') .* eye2))) * ...
% %     lambda / (2 * m);

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
eye1 = eye(size(t1,1));
eye2 = eye(size(t2,1));
J = J+ (sum(sum((t1 * t1') .* eye1)) + sum(sum((t2*t2') .* eye2))) * lambda / (2*m);

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

%  Y 5000 x 10
for i = 1:m
    % part 1
    a1_1 = [1, X(i,:)]'; % 401 x 1
    a2 = sigmoid(Theta1 * a1_1); % 25 x 1
    a2_1 = [1; a2]; %26 x 1
    a3 = sigmoid(Theta2 * a2_1); % 10 x 1
    % part 2
%     delta3 = zeros(num_labels,1); % 10 x 1
    delta3 = a3 - Y(i,:)';
    delta2 = Theta2' * delta3 .* sigmoidGradient([1;Theta1 * a1_1]); % 26 x 1 %***********
    Theta1_grad = Theta1_grad + delta2(2:end) * a1_1'; % 25 x 401
    Theta2_grad = Theta2_grad + delta3 * a2_1'; % 10 x 26
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



Theta1_grad = [Theta1_grad(:,1),Theta1_grad(:,2:end) + lambda * Theta1(:,2:end)/m];
Theta2_grad = [Theta2_grad(:,1),Theta2_grad(:,2:end) + lambda * Theta2(:,2:end)/m];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
