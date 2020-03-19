function f=gaussian_filter(n,s)
%Gaussian filter implemetation/gaussian kernal
%s : S.D
%n: Kernal size
%Author : Achintha Iroshan ,University of Moratuwa

x = -1/2:1/(n-1):1/2;

[Y,X] = meshgrid(x,x);

f = exp( -(X.^2+Y.^2)/(2*s^2) );

f = f / sum(f(:));

end