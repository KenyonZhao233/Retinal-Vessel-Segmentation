function out = matchedFilter(image)
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
    out(im_mask == 0) = 0;
end