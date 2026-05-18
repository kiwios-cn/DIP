function out = my_median(img, win_size)
% 手动中值滤波（不使用内置 medfilt2）
% 输入:
%   img      - 输入图像（灰度 uint8）
%   win_size - 窗口大小，奇数，默认 3
% 输出:
%   out      - 中值滤波后的图像（uint8）

if nargin < 2, win_size = 3; end

half = floor(win_size / 2);
img = double(img);
[H, W] = size(img);

% 边缘填充（复制边缘像素）
img_pad = padarray(img, [half half], 'replicate');

out = zeros(H, W);
for i = 1:H
    for j = 1:W
        patch = img_pad(i:i+win_size-1, j:j+win_size-1);
        out(i,j) = median(patch(:));
    end
end

out = uint8(out);
end
