"""
细胞分割Web应用 - Cellpose深度学习版本 (Flask)
功能：
1. 上传图像进行细胞分割
2. 点击细胞显示放大图和参数
3. 交互式Web界面
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
from skimage import measure
import os

# 设置CPU线程数以提高性能
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量存储Cellpose模型
cellpose_model = None


class CellSegmentationCellposeWeb:
    def __init__(self):
        self.original_image = None
        self.gray_image = None
        self.labeled_image = None
        self.regions = None
        self.model = None

    def load_image(self, image_data):
        """从上传的数据加载图像"""
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # 处理不同的图像格式
        if len(img.shape) == 2:
            # 灰度图像
            self.gray_image = img
            self.original_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            # 彩色图像
            self.original_image = img
            self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return False

        return True

    def load_cellpose_model(self, model_type='cyto'):
        """加载Cellpose模型"""
        global cellpose_model

        if cellpose_model is None:
            try:
                from cellpose import models
                import torch
                torch.set_num_threads(8)
                cellpose_model = models.CellposeModel(gpu=False, model_type=model_type)
            except Exception as e:
                print(f"加载Cellpose模型失败: {e}")
                return False

        self.model = cellpose_model
        return True

    def segment_cells(self, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0):
        """使用Cellpose进行细胞分割"""
        if self.model is None:
            if not self.load_cellpose_model():
                raise RuntimeError("无法加载Cellpose模型")

        result = self.model.eval(
            self.gray_image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0]
        )

        # 兼容不同版本的返回值
        if isinstance(result, tuple):
            if len(result) >= 1:
                masks = result[0]
            else:
                raise RuntimeError("Cellpose返回值格式错误")
        else:
            masks = result

        self.labeled_image = masks
        return self.labeled_image

    def calculate_properties(self, min_area=80, max_area=1000,
                           max_intensity=120, min_circularity=0.20):
        """计算细胞属性并过滤"""
        if self.labeled_image is None:
            return []

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

            # 计算平均灰度
            mean_intensity = getattr(region, 'intensity_mean',
                                   getattr(region, 'mean_intensity', 0))

            # 过滤标准
            if (min_area <= area <= max_area and
                mean_intensity <= max_intensity and
                circularity >= min_circularity):
                filtered_regions.append(region)

        self.regions = filtered_regions
        return self.regions

    def create_result_image(self):
        """创建带轮廓的结果图像"""
        result_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()

        for region in self.regions:
            # 获取轮廓
            contours = measure.find_contours(self.labeled_image == region.label, 0.5)

            # 绘制轮廓
            for contour in contours:
                contour = contour.astype(np.int32)
                for i in range(len(contour)-1):
                    cv2.line(result_image,
                            (contour[i, 1], contour[i, 0]),
                            (contour[i+1, 1], contour[i+1, 0]),
                            (0, 255, 0), 1)

            # 绘制中心点
            y0, x0 = region.centroid
            cv2.circle(result_image, (int(x0), int(y0)), 3, (255, 0, 0), -1)

        return result_image

    def get_cell_info(self, x, y):
        """根据点击坐标获取细胞信息"""
        for region in self.regions:
            # 检查点是否在细胞区域内
            coords = region.coords
            for coord in coords:
                if coord[0] == y and coord[1] == x:
                    return self.get_cell_details(region)
        return None

    def get_cell_details(self, region):
        """获取细胞详细信息"""
        # 长轴长度
        major_axis = getattr(region, 'axis_major_length',
                           getattr(region, 'major_axis_length', 0))

        # 短轴长度
        minor_axis = getattr(region, 'axis_minor_length',
                           getattr(region, 'minor_axis_length', 0))

        # 圆度
        perimeter = region.perimeter
        area = region.area
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # 平均灰度
        mean_intensity = getattr(region, 'intensity_mean',
                               getattr(region, 'mean_intensity', 0))

        # 方差
        coords = region.coords
        pixel_values = [self.gray_image[coord[0], coord[1]] for coord in coords]
        variance = np.var(pixel_values)

        # 创建细胞放大图
        minr, minc, maxr, maxc = region.bbox
        padding = 20
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(self.original_image.shape[0], maxr + padding)
        maxc = min(self.original_image.shape[1], maxc + padding)

        cell_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)[minr:maxr, minc:maxc].copy()

        # 绘制轮廓
        cell_mask = np.zeros((maxr-minr, maxc-minc), dtype=np.uint8)
        for coord in coords:
            r, c = coord[0] - minr, coord[1] - minc
            if 0 <= r < cell_mask.shape[0] and 0 <= c < cell_mask.shape[1]:
                cell_mask[r, c] = 255

        contours = measure.find_contours(cell_mask, 0.5)
        for contour in contours:
            contour = contour.astype(np.int32)
            for i in range(len(contour)-1):
                cv2.line(cell_image,
                        (contour[i, 1], contour[i, 0]),
                        (contour[i+1, 1], contour[i+1, 0]),
                        (0, 255, 0), 1)

        # 转换为base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(cell_image, cv2.COLOR_RGB2BGR))
        cell_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'label': int(region.label),
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'major_axis': float(major_axis),
            'minor_axis': float(minor_axis),
            'mean_intensity': float(mean_intensity),
            'variance': float(variance),
            'centroid': [float(region.centroid[1]), float(region.centroid[0])],
            'bbox': [int(minc), int(minr), int(maxc), int(maxr)],
            'cell_image': cell_image_base64
        }


# 全局分割器对象
segmenter = CellSegmentationCellposeWeb()


@app.route('/')
def index():
    """主页"""
    return render_template('index_cellpose.html')


@app.route('/segment', methods=['POST'])
def segment():
    """执行细胞分割"""
    try:
        # 获取上传的图像
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图像'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        # 获取参数
        diameter = int(request.form.get('diameter', 30))
        flow_threshold = float(request.form.get('flow_threshold', 0.4))
        cellprob_threshold = float(request.form.get('cellprob_threshold', 0.0))
        min_area = int(request.form.get('min_area', 80))
        max_area = int(request.form.get('max_area', 1000))
        max_intensity = int(request.form.get('max_intensity', 120))
        min_circularity = float(request.form.get('min_circularity', 0.20))

        # 加载图像
        image_data = file.read()
        if not segmenter.load_image(image_data):
            return jsonify({'error': '无法加载图像'}), 400

        # 加载模型
        if not segmenter.load_cellpose_model():
            return jsonify({'error': '无法加载Cellpose模型'}), 500

        # 执行分割
        segmenter.segment_cells(diameter, flow_threshold, cellprob_threshold)

        # 计算属性
        segmenter.calculate_properties(min_area, max_area, max_intensity, min_circularity)

        # 创建结果图像
        result_image = segmenter.create_result_image()

        # 转换为base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        # 统计信息
        stats = {
            'cell_count': len(segmenter.regions),
            'avg_area': float(np.mean([r.area for r in segmenter.regions])) if segmenter.regions else 0,
            'avg_circularity': float(np.mean([
                4 * np.pi * r.area / (r.perimeter ** 2) if r.perimeter > 0 else 0
                for r in segmenter.regions
            ])) if segmenter.regions else 0,
            'avg_intensity': float(np.mean([
                getattr(r, 'intensity_mean', getattr(r, 'mean_intensity', 0))
                for r in segmenter.regions
            ])) if segmenter.regions else 0
        }

        return jsonify({
            'success': True,
            'result_image': result_base64,
            'stats': stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_cell_info', methods=['POST'])
def get_cell_info():
    """获取点击位置的细胞信息"""
    try:
        data = request.get_json()
        x = int(data['x'])
        y = int(data['y'])

        cell_info = segmenter.get_cell_info(x, y)

        if cell_info:
            return jsonify({
                'success': True,
                'cell': cell_info
            })
        else:
            return jsonify({
                'success': False,
                'message': '未找到细胞'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
