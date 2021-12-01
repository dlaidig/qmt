function qR = quatRight(q)

if (size(q,1) == 1 && size(q,2) == 4) || (size(q,1) == 4 && size(q,2) == 1)
    qR = [q(1) -q(2) -q(3) -q(4);
          q(2)  q(1)  q(4) -q(3);
          q(3) -q(4)  q(1)  q(2);
          q(4)  q(3) -q(2)  q(1)];
else
    error('Input must be a vector of length 4.')
end