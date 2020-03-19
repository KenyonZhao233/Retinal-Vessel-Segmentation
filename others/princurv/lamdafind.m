function [lamdaplus,lamdaminus]=lamdafind(gxx1,gyy1,gxy1)
%This function perform finding eigen values of hessian matrix and output
%maximum and minimum eigen value as lamdaplus and lambaminus
%gxx1: Second derivative over x
%gyy1: Second deriative over y
%gxy : derivative over x and y 
%Author : Achintha Iroshan ,University of Moratuwa

%Hessian matrix -> H = [gxx gxy;gyx gyy] .Since hessian matrix is 
%symetrical with real eigen values gxy = gyx

H=[gxx1 gxy1;gxy1 gyy1];

%Obtain eigen values
lamda=eig(H);

%Obtain maximum and minimum lamda values
if lamda(1)>lamda(2)
    lamdaplus = lamda(1);
    lamdaminus = lamda(2);
else if lamda(1)<lamda(2)
    lamdaplus = lamda(2);
    lamdaminus = lamda(1);  
    else
        lamdaplus = lamda(1);
        lamdaminus = lamda(2);
    end
end
end