function n = multinorm(v)

N = length(v);
D = find(size(v) == min(size(v)));
n = zeros(N,1);

for k = 1:N
    if D == 1
        n(k) = norm(v(:,k));
    elseif D == 2
        n(k) = norm(v(k,:));
    end
end