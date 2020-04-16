function [im_new, mean_val] = replace_black_ring2(im_enh,im_mask)
%将眼底图像的黑色背景用随机选取的3个(50,50,30)背景矩阵的均值替代
%   输入：
%           im_enh ：对比度增强的图像
%           im_mask：掩模
%   输入：
%           im_new ：替换后的图像
%           mean_val：均值


[row, col] = size(im_mask);
area_sum  = zeros(50,50);     

posit = ceil((rand(3,2)+1)* 1/3*min(row,col));
% figure
for i = 1:3
    x = posit(i,1);y = posit(i,2);
    area_rand= im_enh(x-25:x+24,y-25:y+24); % 随选取背景
    area_sum = area_sum + area_rand;
%     subplot(2,2,i)
%     imshow(area_rand)
end

area_sum = area_sum.*1/3;
% subplot(2,2,4), imshow( area_sum)

mean_val = mean(mean(area_sum));    % 计算每一维的均值
mean_mask = ~im_mask.*mean_val;     % 生成新背景
im_new = mean_mask + im_enh.*im_mask;       % 叠加背景

end

