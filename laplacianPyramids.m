% This code accompanies the paper:
% Real-Time Retinal Vessel Segmentation on High-Resolution Fundus Images Using Laplacian Pyramids
% R. Dachsel, A. J枚ster, M. Breu脽
% Pacific-Rim Symposium on Image and Video Technology 2019
 
% Please cite the paper above if you use the code.

% Written by R. Dachsel
% Brandenburg Technical University 
% (c) 2019



function S = laplacianPyramids(I,plot)

    I=I(:,:,2); %green channel-> best contrast
    
    s = size(I);
    I = padarray(I,[8 - mod(s(1),8),8 - mod(s(2),8)],'pre');
    
    % 掩模生成
    mask = I > 35; 
    mask = imerode(mask, strel('disk',5));
    
    I=double(I);

    %% Parameters
    a=0.3;   % free paramter
    m=512;   %contrast colors
    T=0.10;  % binarization threshold
    k=0.056; %connected components


    %% Laplace Pyramid framework
    [L2,L3,L4]=L_pyramid(I,a,m);
    [S2,S3,S4]=Conn_comp(L2,L3,L4,T,k);
    S=fusion(S2,S3,S4,mask,a);
    if plot == true
        figure();
        subplot(2,2,1);imshow(I,[]);title('绿色通道');
        subplot(2,2,2);imshow(L2,[]);title('滤波后');
        subplot(2,2,3);imshow(S2,[]);title('联通分量处理后');
        subplot(2,2,4);imshow(S,[]);title('细节处理后');
    end
end






