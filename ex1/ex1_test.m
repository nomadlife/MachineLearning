%% Initialization
clear ; close all; clc

%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
%plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;
