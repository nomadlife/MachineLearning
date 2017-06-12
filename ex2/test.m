echo on
data=load('ex2data1.txt');
X=data(:,[1,2]);
y=data(:,3);
pos=find(y==1);
neg=find(y==0);
pause;

%figure;
%hold on;

%graphics_toolkit('fltk');
%plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7);
%plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

%pause;

[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1)
[cost, grad] = costFunction(initial_theta, X, y);


pause;

test_theta = [-24; 0.2; 0.2]
[cost, grad] = costFunction(test_theta, X, y);

pause;

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
	
theta 
cost

prob = sigmoid([1 45 85] * theta);
prob

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


