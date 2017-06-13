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


z=zeros(rows(X));
h=0;
%w=0;
%r=0;
%t=0;

for i=1:m
	z(i)=X(i,:)*theta;
	h=1./(1+e.^-z(i));

	J=J+((-y(i))*log(h)-(1-y(i))*log(1-h))/m;

	%w=w+(h-y(i))*X(i,1)/m;
	%r=r+(h-y(i))*X(i,2)/m;
	%t=t+(h-y(i))*X(i,3)/m;
	
end

for j=1:size(theta,1)
	if (j>=2)
	J=J+(theta(j)^2)*lambda/(2*m);
	end	
end

for j=1:size(theta,1)

	for i=1:m
		z(i)=X(i,:)*theta;
		h=1./(1+e.^-z(i));
		grad(j)=grad(j)+(h-y(i))*X(i,j)/m;
	end
	
	if (j>=2)
		grad(j)=grad(j)+theta(j)*lambda/m;
	end
	
end


%J
%grad=[w,r,t]


% =============================================================

end
