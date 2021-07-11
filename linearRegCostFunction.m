function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hypot=(X*theta);

unreg=(hypot-y).^2; 
unreg=sum(sum(unreg));
unreg=unreg*(1/2/m);

regTerm=sum(sum(theta.^2))-theta(1)^2;
regTerm=regTerm*lambda/2/m;

J=unreg+regTerm;


unreg= (1/m)*(X'*(hypot-y));
regTerm2=(lambda/m)*theta;
regTerm2(1)=0;
grad=unreg+regTerm2;










% =========================================================================

grad = grad(:);

end
