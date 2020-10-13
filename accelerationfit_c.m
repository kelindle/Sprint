function y = accelerationfit(a,x)

y = a(1)*(1-exp(-x./a(2))) + a(3);

