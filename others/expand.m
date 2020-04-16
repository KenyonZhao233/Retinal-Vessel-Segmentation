function I_new=expand(I_old,a)
%% cpu implementation for expand operator
%1d kernel filters for 4 cases
g_uu=[0.25;0.25];                     %filter kernels
g1_ug=[0.25;0.25];
g2_ug=[0.25-a/2;a;0.25-a/2];
g_gg=[0.25-a/2;a;0.25-a/2];

M=size(I_old);
I_new=zeros(2*M(1),2*M(2));           %define sizes
% frame for boundary conditions
A=zeros(M(1)+2, M(2)+2);
N=size(A);
% put image into frame
A(2:N(1)-1, 2:N(2)-1)=I_old;
%left boundary (reflecting BC)
A(1,:)=A(2,:);
%upper boundary
A(:,1)=A(:,2);
%right boundary
A(N(1),:)=A(N(1)-1,:);
%lower boundary
A(:,N(2))=A(:,N(2)-1);
% 2 steb convolution with 1d kernels
% use separability of g-> (g*g)*I= g*(g*f)
%even/even
A1=conv2(A,g_gg,'same');
A2=conv2(A1,g_gg','same');
A3=A2(2:M(1)+1,2:M(2)+1);
I_new(2:2:end,2:2:end)=4*A3;
%odd/odd
A1=conv2(A,g_uu,'same');
A2=conv2(A1,g_uu','same');
A3=A2(1:M(1),1:M(2));
I_new(1:2:end,1:2:end)=4*A3;
%even/odd
A1=conv2(A,g1_ug,'same');
A2=conv2(A1,g2_ug','same');
A3=A2(1:M(1),2:M(2)+1);
I_new(1:2:end,2:2:end)=4*A3;
%odd/even
A1=conv2(A,g2_ug,'same');
A2=conv2(A1,g1_ug','same');
A3=A2(2:M(1)+1,1:M(2));
I_new(2:2:end,1:2:end)=4*A3;
end