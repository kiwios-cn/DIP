function out = my_histeq(img)
% 手动直方图均衡化（不使用内置 histeq）
% 输入:
%   img - 灰度图像（uint8）
% 输出:
%   out - 均衡化后的图像（uint8）

img_d  = double(img);
total  = numel(img_d);

% 统计直方图
h = zeros(1, 256);
for k = 0:255
    h(k+1) = sum(img_d(:) == k);
end

% 累积分布函数 CDF
cdf = cumsum(h);
cdf_min = min(cdf(cdf > 0));

% 计算映射表（查找表 LUT）
% s = round((CDF(r) - CDF_min) / (N - CDF_min) * 255)
map = uint8(floor((cdf - cdf_min) / (total - cdf_min) * 255 + 0.5));

% 像素替换
out = uint8(zeros(size(img_d)));
for k = 0:255
    out(img_d == k) = map(k+1);
end
end
