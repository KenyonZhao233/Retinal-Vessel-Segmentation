function [image_out]=PCAEnhance(image)
%基于PCA增强
I=image;
%I = imread(path);
[len,wid,channel]=size(I);
B = imresize(I, [len/4 wid/4]); 
lab = rgb2lab(im2double(B));
lab(:,:,2)=0;lab(:,:,3)=0;
wlab = reshape(lab,[],3);
[C,S] = pca(wlab); 
S = reshape(S,size(lab));
S = S(:,:,1);
gray = (S-min(S(:)))./(max(S(:))-min(S(:)));
J = adapthisteq(gray,'numTiles',[8 8],'nBins',256); 
h = fspecial('average', [11 11]);
JF = imfilter(J, h);
Z = imsubtract(JF, J); 
level = graythresh(Z);
BW = imbinarize(Z, level-0.008);
BW2 = bwareaopen(BW, 50);
out = imoverlay(B, BW2, [0 0 0]);
image_out=BW2;
