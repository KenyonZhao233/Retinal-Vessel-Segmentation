function im_thre = matchedFilter(image,plot)
    %% 预处理
    % 提取绿色通道
    [~, g, ~] = imsplit(image);
    
    % 掩模生成
    im_mask = g > 40; 
    im_mask = imerode(im_mask, strel('disk',5));
    
    % CLAHE
    bw = adapthisteq(g);

    % 形态学处理
    SE = strel('disk',5);
    %SE = strel('square',5);
    marker = imerode(bw,SE);
    mask = bw;
    mid = imreconstruct(marker,mask);
   
    % 顶帽变换
    mid = imbothat(mid, SE);
    %mid = 255 - mid;
    mid = adapthisteq(mid);
    
    %% Matched Filtering
    img = im2double(mid);
    s = 1.5; %sigma
    L = 7;
    theta = 0:15:360; %different rotations
    out = zeros(size(img));
    m = max(ceil(3*s),(L-1)/2);
    [x,y] = meshgrid(-m:m,-m:m); % non-rotated coordinate system, contains (0,0)
    for t = theta
       t = t / 180 * pi;        % angle in radian
       u = cos(t)*x - sin(t)*y; % rotated coordinate system
       v = sin(t)*x + cos(t)*y; % rotated coordinate system
       N = (abs(u) <= 3*s) & (abs(v) <= L/2); % domain
       k = exp(-u.^2/(2*s.^2)); % kernel
       k = k - mean(k(N));
       k(~N) = 0;               % set kernel outside of domain to 0
       res = conv2(img,k,'same');
       out = max(out,res);
    end
    out = out/max(out(:));
    level = graythresh(out);
    im_thre = imbinarize(out,level) & im_mask;
    if plot == true
        figure();
        subplot(1,5,1);imshow(g,[]);title('绿色通道');
        subplot(1,5,2);imshow(bw,[]);title('CLAHE增强');
        subplot(1,5,3);imshow(mid,[]);title('形态学处理与顶帽变换');
        subplot(1,5,4);imshow(out,[]);title('滤波结果');
        subplot(1,5,5);imshow(im_thre,[]);title('二值化处理');
    end
end