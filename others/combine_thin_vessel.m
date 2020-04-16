function [im_final] = combine_thin_vessel(im_thin_vess,im_sel)
%以细小血管图片（im_thin_vess）为基础，组合顶帽阈值处理后的二值图（im_sel）
%   im_sel中的 1 在im_thin_vess对应位置的 8 领域内1的数量不为0则认为1
%   输入：
%           im_sel ：     粗血管处理结果
%           im_thin_vess：细血管处理结果
%   输入：
%           im_final  ：最终分割结果

[row, col] = size(im_thin_vess);

kernel = [1, 1, 1;
          1, 0, 1;
          1, 1, 1];    
im_final = im_thin_vess;

% 计算对应位置 8 邻域内 1 的数量
for i = 2:row - 1
    for j = 2:col - 1
        if(im_sel(i,j) ~= 0 && sum(sum((im_thin_vess(i-1:i+1,j-1:j+1).*kernel)))> 0)
            im_final(i,j) = im_sel(i,j);
        end   
    end
end

end

