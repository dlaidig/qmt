function [l,dlde] = lossFunctions(e,type,params)

switch type

case 'squared'
    l = norm(e,2)^2;
    dlde = 2*e;
case 'huber'
    if nargin < 3
        delta = 1;
    else
        delta = params(1);
    end
    if norm(e,1) <= delta
        l = norm(e,2)^2/2;
        dlde = e;
    else
        l = delta*(norm(e,1)-delta/2);
        dlde = delta*sign(e);
    end
case 'absolute'
    l = norm(e,1);
    dlde = sign(e);
end
        
        