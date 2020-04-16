function S=fusion(S2,S3,S4,mask,a)

S_2=expand(S2,a);
S_3=expand(expand(S3,a),a);
S_4=expand(expand(expand(S4,a),a),a);

S_all=S_2+S_3+S_4;

S=zeros(size(S_2,1),size(S_2,2));
S(S_all>0.6)=1;

mask=imerode(mask,strel('line',9,1));
mask=double(mask);
S=S.*mask;
end