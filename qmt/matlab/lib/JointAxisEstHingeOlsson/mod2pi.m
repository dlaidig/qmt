function x = mod2pi(x)

for i = 1:size(x,1)
    for j = 1:size(x,2);
        while abs(x(i,j)) > pi
            if x(i,j) > 0
                x(i,j) = x(i,j) - 2*pi;
            else
                x(i,j) = x(i,j) + 2*pi;
            end 
        end
    end
end