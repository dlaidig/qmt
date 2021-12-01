function X = crossMat(x)

if length(x) == 3
    X = [0   -x(3) x(2);
         x(3) 0   -x(1);
        -x(2) x(1) 0];
else
    X = zeros(3);
end
        