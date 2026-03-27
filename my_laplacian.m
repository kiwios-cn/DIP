function out = my_laplacian(img, k)
% 拉普拉斯锐化增强
% 输入:
%   img - 灰度图像（uint8）
%   k   - 增强系数，默认 1.0
% 输出:
%   out - 锐化后的图像（uint8）

if nargin < 2, k = 1.0; end

% 拉普拉斯卷积核
kernel = [0 -1  0;
         -1  4 -1;
          0 -1  0];

img_d = double(img);
lap   = conv2(img_d, kernel, 'same');

% 增强 + 截断到 [0,255] + 转回 uint8
out = uint8(min(max(img_d - k * lap, 0), 255));
end
