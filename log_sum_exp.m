function l_sum = log_sum_exp(v)

% l_sum = log_sum_exp(v, dim)
%
% This function calculates the logarithm of the sum of the numbers whose
% logarithms are given in the vector v in a stable way. It does the
% summation by extracting the maximum of the numbers' logarithms so that
% the arguement of the exponential function is small enough to be safe.
%
% This is the simple version of log_sum
%
% Sinan Yildirim, 20.11.2018

m = max(v);
l_sum = log(sum(exp(v- m))) + m;