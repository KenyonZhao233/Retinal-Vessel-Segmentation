function [L2,L3,L4]=L_pyramid(I_old,a,m)

G1=reduce(I_old,a);
G2=reduce(G1,a);
G3=reduce(G2,a);
G4=reduce(G3,a);

%G10=expand(G1,a);
G11=expand(G2,a);
G12=expand(G3,a);
G13=expand(G4,a);


%L1= I_old - G10;
L2= G1 - G11;
L3= G2 - G12;
L4= G3 - G13;

x=L2; % scale image for more contrast
xmin = min(min(x));
xmax = max(max(x));
x = round((m-1)*(x-xmin)/(xmax-xmin));
f = find(diff(sort([x(:); (0:m)'])));
f = f/max(f);
L_sc=f(x+1,1);
L2= reshape(L_sc,size(x,1),size(x,2),1);


x=L3; % scale image for more contrast
xmin = min(min(x));
xmax = max(max(x));
x = round((m-1)*(x-xmin)/(xmax-xmin));
f = find(diff(sort([x(:); (0:m)'])));
f = f/max(f);
L_sc=f(x+1,1);
L3= reshape(L_sc,size(x,1),size(x,2),1);


x=L4; % scale image for more contrast
xmin = min(min(x));
xmax = max(max(x));
x = round((m-1)*(x-xmin)/(xmax-xmin));
f = find(diff(sort([x(:); (0:m)'])));
f = f/max(f);
L_sc=f(x+1,1);
L4= reshape(L_sc,size(x,1),size(x,2),1);



end
