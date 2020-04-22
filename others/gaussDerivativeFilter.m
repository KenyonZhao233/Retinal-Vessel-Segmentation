function im_final = gaussDerivativeFilter(image,plot)
%视网膜血管提取函数
%   血管提取主要分为四部分：预处理、粗血管提取、细血管提取、后处理；

%% 预处理
im_rgb = im2double(image);
% 掩模生成
% im_mask = im_rgb(:,:,2) > (20/255);    % For DRIVE
im_mask = im_rgb(:,:,2) > (40/255);  % For STARE
im_mask = double(imerode(im_mask, strel('disk',3)));
% 提取新的绿色通道
im_green = im_rgb(:,:,2);
% 对比度增强 CLAHE
im_enh = adapthisteq(im_green,'numTiles',[8 8],'nBins',128);

%% 处理一
% 替换黑色背景
[im_enh1, mean_val] = replace_black_ring2(im_enh,im_mask);
im_gray = imcomplement(im_enh1); 
% 顶帽变换
se = strel('disk',10);
im_top = imtophat(im_gray,se);  
% OTSU 阈值处理
level = graythresh(im_top);
im_thre = imbinarize(im_top,level) & im_mask;
% 删除小面积对象
im_rmpix = bwareaopen(im_thre,100,8);
% 根据绿色通道增强与阈值处理的结果，去除部分非血管像素
[im_sel] = vessel_point_selected(im_gray,im_rmpix,mean_val);
%% 处理二
im_thin_vess = MatchFilterWithGaussDerivative(im_enh, 1, 4, 12, im_mask, 2.3, 30);

%% 后处理
% 合并处理一处理二结果
[im_final] = combine_thin_vessel(im_thin_vess,im_sel);
if plot == true
    figure();
    subplot(1,5,1);imshow(im_green,[]);title('绿色通道');
    subplot(1,5,2);imshow(im_enh,[]);title('CLAHE增强');
    subplot(1,5,3);imshow(im_top,[]);title('替换背景与顶帽变换');
    subplot(1,5,4);imshow(im_thre,[]);title('OTSU 阈值处理');
    subplot(1,5,5);imshow(im_final,[]);title('细节处理');
end
end

