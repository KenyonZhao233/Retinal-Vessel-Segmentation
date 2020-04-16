function [S2,S3,S4]=Conn_comp(L2,L3,L4,T,k)

[counts,loc]=imhist(L2);
count2=cumsum(counts)/(size(L2,1)*size(L2,2));
T2=max(loc(count2 < T));
L_bw=zeros(size(L2,1),size(L2,2)); %binarize image
L_bw(L2<=T2)=255;

P=k*sqrt(size(L2,1)*size(L2,2));
label=bwlabel(L_bw);
label_size = regionprops(label, 'Area');
S2= ismember(label, find([label_size.Area] >=P));

[counts,loc]=imhist(L3);
count2=cumsum(counts)/(size(L3,1)*size(L3,2));
T2=max(loc(count2 < T2));
L_bw=zeros(size(L3,1),size(L3,2)); %binarize image
L_bw(L3<=T2)=255;

P=k*sqrt(size(L3,1)*size(L3,2));
label=bwlabel(L_bw);
label_size = regionprops(label, 'Area');
S3= ismember(label, find([label_size.Area] >=P));

[counts,loc]=imhist(L4);
count2=cumsum(counts)/(size(L4,1)*size(L4,2));
T2=max(loc(count2 < T2));
L_bw=zeros(size(L4,1),size(L4,2)); %binarize image
L_bw(L4<=T2)=255;

P=k*sqrt(size(L4,1)*size(L4,2));
label=bwlabel(L_bw);
label_size = regionprops(label, 'Area');
S4= ismember(label, find([label_size.Area] >=P));

end