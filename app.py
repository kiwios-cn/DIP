"""
细胞分割Web应用 - Flask后端
功能：
1. 上传图像进行细胞分割
2. 点击细胞显示放大图和参数
3. 交互式Web界面
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
from skimage import measure, morphology, filters
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import heapq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


class CellSegmentationWeb:
    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.labeled_image = None
        self.regions = None
        self.preprocessed = None

    def load_image(self, image_data):
        """从上传的数据加载图像"""
        nparr = np.frombuffer(image_data, np.uint8)
        self.original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.original_image is not None

    def preprocess(self):
        """预处理图像"""
        bilateral = cv2.bilateralFilter(self.gray_image, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.preprocessed = clahe.apply(bilateral)
        return self.preprocessed

    def manual_watershed(self, gradient, markers, mask):
        """
        手动实现分水岭算法

        参数:
            gradient: 梯度图像（地形图），像素值代表高度
            markers: 标记图像，每个种子点有唯一的正整数标签
            mask: 二值掩码，指定分割区域

        返回:
            labels: 分割结果，每个区域有唯一标签

        算法原理:
            1. 将图像看作地形图，像素值代表高度
            2. 从标记点（种子点）开始，模拟水淹过程
            3. 使用优先队列按照高度从低到高处理像素
            4. 当不同区域的水相遇时，形成分水岭边界
        """
        h, w = gradient.shape
        labels = markers.copy().astype(np.int32)

        # 分水岭边界标记为-1
        WATERSHED_BOUNDARY = -1

        # 优先队列：(高度, y坐标, x坐标)
        pq = []

        # 初始化：将所有标记点的邻居加入优先队列
        for y in range(h):
            for x in range(w):
                if labels[y, x] > 0:  # 已标记的种子点
                    # 检查8邻域
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue

                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] > 0 and labels[ny, nx] == 0:
                                    # 未标记的邻居加入队列
                                    heapq.heappush(pq, (int(gradient[ny, nx]), ny, nx))

        # 分水岭主循环：按高度从低到高处理像素
        while pq:
            height, y, x = heapq.heappop(pq)

            # 如果该像素已被标记，跳过
            if labels[y, x] != 0:
                continue

            # 检查邻域，找到该像素应该属于的区域
            neighbor_labels = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if labels[ny, nx] > 0:  # 已标记的邻居
                            neighbor_labels.append(labels[ny, nx])

            if len(neighbor_labels) == 0:
                # 没有已标记的邻居，跳过
                continue

            # 检查邻居标签是否一致
            unique_labels = set(neighbor_labels)

            if len(unique_labels) == 1:
                # 所有邻居属于同一区域，该像素也属于该区域
                labels[y, x] = neighbor_labels[0]

                # 将该像素的未标记邻居加入队列
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] > 0 and labels[ny, nx] == 0:
                                heapq.heappush(pq, (int(gradient[ny, nx]), ny, nx))

            else:
                # 邻居属于不同区域，该像素是分水岭边界
                labels[y, x] = WATERSHED_BOUNDARY

        # 将分水岭边界(-1)转换为0（背景）
        labels[labels == WATERSHED_BOUNDARY] = 0

        return labels

    def segment_cells(self):
        """细胞分割"""
        preprocessed = self.preprocess()

        # Otsu阈值
        _, binary_otsu = cv2.threshold(preprocessed, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 自适应阈值
        binary_adaptive = cv2.adaptiveThreshold(preprocessed, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 21, 5)

        # 结合两种方法
        binary = cv2.bitwise_and(binary_otsu, binary_adaptive)

        # 形态学操作（减少开运算迭代以保留更多边缘）
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        # 去除边界对象
        binary = self.remove_border_objects(binary)

        # 去除小对象
        binary = self.remove_small_objects(binary, min_size=100)

        # 距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # 局部最大值（第一阶段：适度提高参数减少误分割）
        local_max = peak_local_max(dist_transform, min_distance=15,
                                   threshold_rel=0.15, labels=binary)

        # 创建标记
        markers = np.zeros_like(binary, dtype=np.int32)
        for idx, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = idx

        # 扩展标记（减小扩展半径以避免过度合并）
        markers = morphology.dilation(markers, morphology.disk(2))

        # 梯度图像
        gradient = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 手动实现的分水岭算法
        markers = self.manual_watershed(gradient, markers, binary)

        self.labeled_image = markers
        return self.labeled_image

    def remove_border_objects(self, binary):
        """去除接触边界的对象"""
        labeled = measure.label(binary)
        border_labels = set()
        h, w = binary.shape

        border_labels.update(labeled[0, :])
        border_labels.update(labeled[-1, :])
        border_labels.update(labeled[:, 0])
        border_labels.update(labeled[:, -1])
        border_labels.discard(0)

        mask = np.ones_like(binary)
        for label in border_labels:
            mask[labeled == label] = 0

        return binary * mask

    def remove_small_objects(self, binary, min_size=100):
        """去除小对象"""
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        mask = np.zeros_like(binary)
        for region in regions:
            if region.area >= min_size:
                coords = region.coords
                for coord in coords:
                    mask[coord[0], coord[1]] = 1

        return binary * mask

    def merge_small_fragments(self, labels_image):
        """第二阶段：合并小碎片到相邻区域"""
        merged_labels = labels_image.copy()
        regions = measure.regionprops(labels_image, intensity_image=self.gray_image)

        # 找出所有小碎片（面积<100）
        small_fragments = []
        region_dict = {}  # label -> region

        for region in regions:
            region_dict[region.label] = region
            if region.area < 100:
                small_fragments.append(region)

        merged_count = 0
        for fragment in small_fragments:
            fragment_label = fragment.label
            fragment_intensity = getattr(fragment, 'intensity_mean',
                                        getattr(fragment, 'mean_intensity', 0))

            # 找到相邻区域
            fragment_mask = (labels_image == fragment_label)
            dilated = morphology.binary_dilation(fragment_mask, morphology.disk(1))

            # 找到膨胀区域内的其他标签
            neighbor_labels = np.unique(labels_image[dilated])
            neighbor_labels = neighbor_labels[neighbor_labels != fragment_label]
            neighbor_labels = neighbor_labels[neighbor_labels != 0]

            if len(neighbor_labels) == 0:
                continue

            # 找到最佳合并候选（灰度最接近的邻居）
            best_neighbor = None
            min_intensity_diff = float('inf')

            for neighbor_label in neighbor_labels:
                if neighbor_label not in region_dict:
                    continue

                neighbor_region = region_dict[neighbor_label]
                neighbor_intensity = getattr(neighbor_region, 'intensity_mean',
                                            getattr(neighbor_region, 'mean_intensity', 0))

                intensity_diff = abs(fragment_intensity - neighbor_intensity)
                combined_area = fragment.area + neighbor_region.area

                # 检查合并条件：灰度差<20 且合并后总面积<1000
                if intensity_diff < 20 and combined_area < 1000:
                    if intensity_diff < min_intensity_diff:
                        min_intensity_diff = intensity_diff
                        best_neighbor = neighbor_label

            # 执行合并
            if best_neighbor is not None:
                merged_labels[fragment_mask] = best_neighbor
                merged_count += 1

        return merged_labels

    def calculate_properties(self):
        """计算细胞属性"""
        if self.labeled_image is None:
            self.segment_cells()

        # 第二阶段：合并小碎片
        self.labeled_image = self.merge_small_fragments(self.labeled_image)

        self.regions = measure.regionprops(self.labeled_image,
                                          intensity_image=self.gray_image)

        # 过滤不符合标准的细胞
        filtered_regions = []
        for region in self.regions:
            area = region.area
            perimeter = region.perimeter

            # 计算圆度
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0

            # 计算平均灰度（兼容不同版本的API）
            mean_intensity = getattr(region, 'intensity_mean',
                                   getattr(region, 'mean_intensity', 0))

            # 过滤标准（混合策略 + 背景过滤）：
            # - 面积: 80 ~ 1000 像素²
            # - 平均灰度: <= 120 (过滤高灰度背景区域)
            # - 圆度: >= 0.20
            if (80 <= area <= 1000 and
                mean_intensity <= 120 and
                circularity >= 0.20):
                filtered_regions.append(region)

        self.regions = filtered_regions
        return self.regions

    def get_annotated_image(self):
        """获取带标注的图像（只有边界框）"""
        if self.regions is None:
            self.calculate_properties()

        # 复制原图
        annotated = self.original_image.copy()

        # 为每个细胞绘制边界框
        for region in self.regions:
            # 获取边界框
            minr, minc, maxr, maxc = region.bbox

            # 绘制矩形框（绿色，细线）
            cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (0, 255, 0), 1)

        return annotated

    def get_cell_info(self, x, y):
        """根据点击坐标获取细胞信息"""
        if self.labeled_image is None or self.regions is None:
            return None

        # 检查坐标是否在图像范围内
        if (x < 0 or x >= self.labeled_image.shape[1] or
            y < 0 or y >= self.labeled_image.shape[0]):
            return None

        # 获取标签
        label = self.labeled_image[y, x]

        if label == 0:
            return None

        # 找到对应的region
        region = None
        for r in self.regions:
            if r.label == label:
                region = r
                break

        if region is None:
            return None

        # 计算参数
        major_axis = getattr(region, 'axis_major_length',
                           getattr(region, 'major_axis_length', 0))
        minor_axis = getattr(region, 'axis_minor_length',
                           getattr(region, 'minor_axis_length', 0))

        perimeter = region.perimeter
        area = region.area
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        mean_intensity = getattr(region, 'intensity_mean',
                               getattr(region, 'mean_intensity', 0))

        coords = region.coords
        pixel_values = [self.gray_image[coord[0], coord[1]] for coord in coords]
        variance = np.var(pixel_values)

        # 提取细胞区域图像（放大）
        minr, minc, maxr, maxc = region.bbox
        padding = 20
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(self.original_image.shape[0], maxr + padding)
        maxc = min(self.original_image.shape[1], maxc + padding)

        cell_crop = self.original_image[minr:maxr, minc:maxc].copy()

        # 在裁剪图上绘制轮廓
        mask_crop = (self.labeled_image[minr:maxr, minc:maxc] == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cell_crop, contours, -1, (0, 255, 0), 1)

        # 转换为base64
        _, buffer = cv2.imencode('.png', cell_crop)
        cell_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'label': int(label),
            'major_axis': float(major_axis),
            'minor_axis': float(minor_axis),
            'circularity': float(circularity),
            'area': float(area),
            'perimeter': float(perimeter),
            'mean_intensity': float(mean_intensity),
            'variance': float(variance),
            'cell_image': cell_image_base64,
            'bbox': [int(minc), int(minr), int(maxc), int(maxr)]
        }


# 全局分割器实例
segmenter = CellSegmentationWeb()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """上传并处理图像"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    try:
        # 读取图像
        image_data = file.read()
        if not segmenter.load_image(image_data):
            return jsonify({'error': '无法读取图像'}), 400

        # 分割细胞
        segmenter.segment_cells()
        segmenter.calculate_properties()

        # 获取带标注的图像
        annotated_image = segmenter.get_annotated_image()

        # 转换为base64
        _, buffer = cv2.imencode('.png', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_base64,
            'cell_count': len(segmenter.regions),
            'width': annotated_image.shape[1],
            'height': annotated_image.shape[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_cell_info', methods=['POST'])
def get_cell_info():
    """获取点击位置的细胞信息"""
    data = request.json
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))

    cell_info = segmenter.get_cell_info(x, y)

    if cell_info is None:
        return jsonify({'error': '未找到细胞'}), 404

    return jsonify(cell_info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
