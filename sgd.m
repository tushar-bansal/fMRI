function [wFinal, J_history] = sgd(X, y, w, lambda, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
n = length(y); % number of training examples
J_history = zeros(num_iters, 1);
wFinal = w;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


j = randi([1 size(X,2)],1,1);
s = (X(:,j)'*(X*wFinal - y))/n + lambda;
wFinal(j)  = wFinal(j) + max(-wFinal(j),-s);

    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, wFinal,lambda);

end


end
