function lamda2=prinCur(Image)
%This Function perform region extraction of feature - principal curvature
%of the image which is needed in applying region groeing
%Image : Input Image
%Here we obtain parameters for Hessian metrx in every pixel and find eigen
%values of the hessian matrix using lamdafind function
%Author : Achintha Iroshan ,University of Moratuwa

% Obtain parameters for hessien matrix
[gx, gy] = gradient(double(Image));
[gxx, gxy] = gradient(gx);
[gxy, gyy] = gradient(gy);



[row,col]=size(Image);
lamdaplus = zeros(row,col);
lamdaminus = zeros(row,col);

%finding eigen values of hessian matrix [gxx gxy;gxy gyy]

for r = 1:row
    for c = 1:col
            [lamdaplus(r,c),lamdaminus(r,c)]=lamdafind(gxx(r,c),gyy(r,c),gxy(r,c));
    end
end


%lamda1=min(abs(lamdaplus),abs(lamdaminus));

%obtain the maximum principal curvature 
%lamda2=max(abs(lamdaplus),abs(lamdaminus));
lamda2 = lamdaplus;
end