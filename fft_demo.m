% FFT 图像频域分析演示
% 包含：旋转、平移、尺度变换、直流分量、低中高频编辑
%
% 使用方法：
%   im = imread('your_image.jpg');
%   im = im2double(rgb2gray(im));
%   运行各 Section 查看效果

%% 公共准备
% im = im2double(rgb2gray(imread('your_image.jpg')));
% [H, W] = size(im);
% F = fftshift(fft2(im));

%% Section 1 - 旋转性质
% 空间域旋转 θ° ↔ 频域幅度谱同步旋转 θ°
function demo_rotation(im, angle_deg)
    F1 = fftshift(fft2(im));
    im_rot = imrotate(im, angle_deg, 'crop');   % 'crop' 保持尺寸不变
    F2 = fftshift(fft2(im_rot));

    figure;
    subplot(2,2,1); imshow(im);           title('原图');
    subplot(2,2,2); imshow(im_rot);       title(sprintf('旋转 %d°', angle_deg));
    subplot(2,2,3); imshow(mat2gray(log(1+abs(F1)))); title('原图频谱');
    subplot(2,2,4); imshow(mat2gray(log(1+abs(F2)))); title('旋转后频谱');
end

%% Section 2 - 平移性质
% 空间域平移 ↔ 频域幅度不变，相位线性变化
function demo_translation(im)
    F1 = fftshift(fft2(im));
    im_shift = circshift(im, [30, 50]);   % 循环移位（与FFT周期性一致）
    F2 = fftshift(fft2(im_shift));

    mag1 = log(1+abs(F1)); mag2 = log(1+abs(F2));
    pha1 = angle(F1);       pha2 = angle(F2);
    fprintf('幅度最大差异: %.6f（应接近0）\n', max(abs(mag1(:)-mag2(:))));

    figure;
    subplot(2,2,1); imshow(mat2gray(mag1)); title('原图幅度谱');
    subplot(2,2,2); imshow(mat2gray(mag2)); title('平移后幅度谱');
    subplot(2,2,3); imshow(mat2gray(pha1)); title('原图相位谱');
    subplot(2,2,4); imshow(mat2gray(pha1-pha2)); title('相位差（线性变化）');
end

%% Section 3 - 尺度变换
% 空间域放大 ↔ 频域压缩（低频集中）
% 空间域缩小 ↔ 频域扩展（高频增多）
function demo_scale(im)
    [H, W] = size(im);
    F1 = fftshift(fft2(im));

    im_big = imresize(im, 2.0);
    im_big = im_big(1:H, 1:W);           % 裁剪回原尺寸（必须！）
    F2 = fftshift(fft2(im_big));

    im_small = imresize(im, 0.5);
    im_small = imresize(im_small, [H, W]); % 插值还原（有损）
    F3 = fftshift(fft2(im_small));

    figure;
    subplot(2,3,1); imshow(im);           title('原图');
    subplot(2,3,2); imshow(im_big);       title('放大×2（裁剪）');
    subplot(2,3,3); imshow(im_small);     title('缩小×0.5（还原）');
    subplot(2,3,4); imshow(mat2gray(log(1+abs(F1)))); title('原图频谱');
    subplot(2,3,5); imshow(mat2gray(log(1+abs(F2)))); title('放大频谱（压缩）');
    subplot(2,3,6); imshow(mat2gray(log(1+abs(F3)))); title('缩小频谱（扩展）');
end

%% Section 4 - 直流分量
% 直流 F(0,0) = 所有像素均值 × 像素总数，代表整体亮度
function demo_dc(im)
    F = fftshift(fft2(im));
    cx = floor(size(F,1)/2)+1;
    cy = floor(size(F,2)/2)+1;

    dc_orig = F(cx, cy);
    F1 = F; F1(cx,cy) = 0;           % 去除直流
    F2 = F; F2(cx,cy) = dc_orig * 2; % 直流×2（整体变亮）
    F3 = F; F3(cx,cy) = dc_orig * 0.2; % 直流×0.2（整体变暗）

    im1 = mat2gray(real(ifft2(ifftshift(F1))));  % 必须 ifftshift 再 ifft2
    im2 = mat2gray(real(ifft2(ifftshift(F2))));
    im3 = mat2gray(real(ifft2(ifftshift(F3))));

    figure;
    subplot(2,2,1); imshow(im);  title('原图');
    subplot(2,2,2); imshow(im1); title('去直流');
    subplot(2,2,3); imshow(im2); title('直流×2（变亮）');
    subplot(2,2,4); imshow(im3); title('直流×0.2（变暗）');
end

%% Section 5/6/7 - 低/中/高频编辑
% 低频（<8%）：整体亮度、大轮廓
% 中频（8~25%）：主要纹理结构
% 高频（>25%）：边缘细节
function demo_freq_edit(im)
    [H, W] = size(im);
    F = fftshift(fft2(im));

    cx = floor(H/2)+1; cy = floor(W/2)+1;
    [u, v] = meshgrid(1:W, 1:H);
    R = sqrt((v-cx).^2 + (u-cy).^2);   % 各点到中心距离

    r_low  = min(H,W) * 0.08;  % 低频半径 8%
    r_mid  = min(H,W) * 0.25;  % 中频上限 25%

    mask_low  = (R <= r_low);
    mask_mid  = (R > r_low) & (R <= r_mid);
    mask_high = (R > r_mid);

    % --- 去除各频段 ---
    F1 = F; F1(mask_low)  = 0;
    F2 = F; F2(mask_mid)  = 0;
    F3 = F; F3(mask_high) = 0;

    % --- 只保留各频段 ---
    F4 = F; F4(~mask_low)  = 0;
    F5 = F; F5(~mask_mid)  = 0;
    F6 = F; F6(~mask_high) = 0;

    % ifftshift → ifft2 取实部（顺序不能颠倒！）
    to_img = @(Fx) mat2gray(real(ifft2(ifftshift(Fx))));

    figure('Name','去除各频段');
    subplot(1,3,1); imshow(to_img(F1)); title('去低频');
    subplot(1,3,2); imshow(to_img(F2)); title('去中频');
    subplot(1,3,3); imshow(to_img(F3)); title('去高频（低通滤波）');

    figure('Name','只保留各频段');
    subplot(1,3,1); imshow(to_img(F4)); title('只保留低频');
    subplot(1,3,2); imshow(to_img(F5)); title('只保留中频');
    subplot(1,3,3); imshow(to_img(F6)); title('只保留高频（边缘）');
end
