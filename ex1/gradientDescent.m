function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%1st solution
%d1=0;d2=0;
%for i=1:m
%d1=d1+(X(i,:)*theta-y(i))*alpha/m;
%d2=d2+(X(i,:)*theta-y(i))*X(i,2)*alpha/m;
%end
%theta=[theta(1)-d1;theta(2)-d2];

%2nd solution.
%theta=theta-transpose(sum((X*theta-y).*[ones(size(X),1),X(:,2)]))*alpha/m;

%3rd solution.
%theta=theta-[ones(1,size(X));X'(2,:)]*(X*theta-y)*alpha/m;

%4th solution.
%theta-=[ones(1,size(X));X'(2,:)]*(X*theta-y)*alpha/m;

%5th solution
theta-=X'*(X*theta-y)*alpha/m;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
