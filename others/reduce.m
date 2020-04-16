function I_new=reduce(I_old,a)
%% cpu implementation for reduce operator
%1d kernel filter
g=[0.25-a/2;0.25;a;0.25;0.25-a/2];
M=size(I_old);                        %define sizes
% frame for boundary conditions
A=zeros(M(1)+4, M(2)+4);
N=size(A);
% put image into frame
A(3:N(1)-2, 3:N(2)-2)=I_old;                    %boundary conditions
%left boundary (reflecting BC)
A(1,:)=A(4,:);
A(2,:)=A(3,:);
%upper boundary
A(:,1)=A(:,4);
A(:,2)=A(:,3);
%right boundary
A(N(1),:)=A(N(1)-3,:);
A(N(1)-1,:)=A(N(1)-2,:);
%lower boundary
A(:,N(2))=A(:,N(2)-3);
A(:,N(2)-1)=A(:,N(2)-2);
% 2 steb convolution with 1d kernel
% use separability of g-> (g*g)*I= g*(g*f)
A1 = conv2(A,g,'same');                         %convolution
A2 = conv2(A1,g','same');
%remove frame + upsampling
I_new= A2(4:2:end-2,4:2:end-2);                 %upsampling
end