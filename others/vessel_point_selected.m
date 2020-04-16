function [im_sel] = vessel_point_selected(im_gray,im_thre,mean_val)
%根据灰度图像与阈值处理的图像的进一步优化
%   对非 0 像素位置对应的灰度图像以及阈值处理结果像素点的距离
%  （像素值与背景或者血管像素的差），进一步决定血管
%   输入：
%           im_gray ：灰度图像
%           im_thre ：阈值处理图像
%           mean_val：背景均值
%   输入：
%           im_sel  ：优化图像

[row, col] = size(im_gray);
im_sel = zeros(row, col);

p_max = max(max(im_gray));
p_min = mean_val;

for i = 1:row
    for j = 1:col
        if(im_thre(i,j) ~= 0)
            if(abs(im_gray(i,j)-p_max) < abs(im_gray(i,j)-p_min))
                % vessel pixel
                im_sel(i,j) = 1;
            end            
        end   
    end
end

% figure, imshow(im_sel),title('im pixel selected');
im_med = medfilt2(im_thre,[3,3]); 
im_sel = im_sel| im_med;
% figure, imshow(im_sel),title('im pixel selected');

end

