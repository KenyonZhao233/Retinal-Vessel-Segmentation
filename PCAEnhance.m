function [image_out]=pcaEnhance(I,plot)
%基于PCA增强的算法
%转换到lab空间
lab = rgb2lab(im2double(I));
lab(:,:,2)=0;lab(:,:,3)=0;
wlab = reshape(lab,[],3);%向量化
[C,S] = pca(wlab); %主成分分析
S = reshape(S,size(lab));%S为PCA后新坐标下的矩阵
S = S(:,:,1);
gray = (S-min(S(:)))./(max(S(:))-min(S(:)));%归一化
J = adapthisteq(gray,'numTiles',[8 8],'nBins',256); %CLAHE直方图均衡
h = fspecial('average', [11 11]);%创建平均滤波算子
JF = imfilter(J, h);%滤波处理
Z = imsubtract(JF, J);% 取灰度图像与平均滤波的差值
level = graythresh(Z);%找到图片的一个合适的阈值
BW = imbinarize(Z, level-0.008);%灰度图转为二进制
BW2 = bwareaopen(BW, 50);%删除二值图像BW中面积小于50的对象
image_out=BW2;
if plot==true
    %CLAHE：JF
    %CLAHE滤波后差值：Z
    %结果：image_out
    figure(1),
    subplot(131),imshow(JF),title('CLAHE后均值滤波');
    subplot(132),imshow(Z),title('CLAHE后均值滤波差值');
    subplot(133),imshow(image_out),title('最终结果');
end
