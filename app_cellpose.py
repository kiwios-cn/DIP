"""
细胞分割Web应用 - Cellpose深度学习版本
使用Cellpose预训练模型进行细胞分割
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import io

# 设置页面配置
st.set_page_config(
    page_title="细胞分割系统 - Cellpose版",
    page_icon="🔬",
    layout="wide"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CellSegmentationCellpose:
    def __init__(self, image):
        """
        初始化细胞分割对象

        参数:
        - image: PIL Image对象或numpy数组
        """
        if isinstance(image, Image.Image):
            self.original_image = np.array(image)
        else:
            self.original_image = image

        # 转换为BGR格式（OpenCV格式）
        if len(self.original_image.shape) == 3:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)

        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.labeled_image = None
        self.regions = None
        self.model = None

    def load_cellpose_model(self, model_type='cyto', gpu=False):
        """加载Cellpose模型"""
        try:
            from cellpose import models
            if self.model is None:
                with st.spinner(f'正在加载Cellpose模型 ({model_type})...'):
                    self.model = models.CellposeModel(gpu=gpu, model_type=model_type)
            return True
        except ImportError:
            st.error("错误: 未安装cellpose库")
            st.info("请运行: pip install cellpose")
            return False
        except Exception as e:
            st.error(f"加载模型失败: {e}")
            return False

    def segment_cells(self, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0):
        """
        使用Cellpose进行细胞分割

        参数:
        - diameter: 预期细胞直径（像素）
        - flow_threshold: 流场阈值
        - cellprob_threshold: 细胞概率阈值
        """
        if self.model is None:
            if not self.load_cellpose_model():
                raise RuntimeError("无法加载Cellpose模型")

        with st.spinner('正在使用Cellpose分割细胞...'):
            result = self.model.eval(
                self.gray_image,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=[0, 0]  # [0,0]表示灰度图像
            )

            # 兼容不同版本的返回值
            if len(result) == 4:
                masks, flows, styles, diams = result
            else:
                masks, flows, styles = result
                diams = diameter  # 使用输入的直径

        self.labeled_image = masks
        st.success(f"Cellpose检测到 {masks.max()} 个区域")
        if isinstance(diams, (int, float)):
            st.info(f"估计的细胞直径: {diams:.1f} 像素")
        else:
            st.info(f"使用的细胞直径: {diameter} 像素")

        return self.labeled_image

    def calculate_properties(self, min_area=80, max_area=1000,
                           max_intensity=120, min_circularity=0.20):
        """
        计算细胞属性并过滤

        参数:
        - min_area: 最小面积
        - max_area: 最大面积
        - max_intensity: 最大灰度值（过滤背景）
        - min_circularity: 最小圆度
        """
        if self.labeled_image is None:
            self.segment_cells()

        from skimage import measure

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

    def get_cell_parameters(self, region):
        """获取单个细胞的参数"""
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

        # 细胞内像素均值
        mean_intensity = getattr(region, 'intensity_mean',
                               getattr(region, 'mean_intensity', 0))

        # 细胞内像素方差
        coords = region.coords
        pixel_values = [self.gray_image[coord[0], coord[1]] for coord in coords]
        variance = np.var(pixel_values)

        return {
            '长轴长度': major_axis,
            '短轴长度': minor_axis,
            '圆度': circularity,
            '均值': mean_intensity,
            '方差': variance,
            '面积': area,
            '周长': perimeter
        }

    def create_segmentation_image(self):
        """创建分割结果图像"""
        from skimage import measure

        # 创建结果图像
        result_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()

        # 绘制每个细胞的轮廓和中心点
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

    def create_cell_detail_image(self, region):
        """创建单个细胞的详细图像"""
        from skimage import measure

        # 获取细胞的边界框
        minr, minc, maxr, maxc = region.bbox

        # 扩展边界框
        padding = 20
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(self.original_image.shape[0], maxr + padding)
        maxc = min(self.original_image.shape[1], maxc + padding)

        # 裁剪图像
        cell_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)[minr:maxr, minc:maxc].copy()

        # 创建mask
        cell_mask = np.zeros((maxr-minr, maxc-minc), dtype=np.uint8)
        coords = region.coords
        for coord in coords:
            r, c = coord[0] - minr, coord[1] - minc
            if 0 <= r < cell_mask.shape[0] and 0 <= c < cell_mask.shape[1]:
                cell_mask[r, c] = 255

        # 找到轮廓
        contours = measure.find_contours(cell_mask, 0.5)

        # 绘制轮廓
        for contour in contours:
            contour = contour.astype(np.int32)
            for i in range(len(contour)-1):
                cv2.line(cell_image,
                        (contour[i, 1], contour[i, 0]),
                        (contour[i+1, 1], contour[i+1, 0]),
                        (0, 255, 0), 1)

        return cell_image


@st.cache_resource
def load_cellpose_model_cached(model_type='cyto'):
    """缓存Cellpose模型加载"""
    try:
        from cellpose import models
        return models.CellposeModel(gpu=False, model_type=model_type)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


def main():
    st.title("🔬 细胞分割系统 - Cellpose深度学习版")
    st.markdown("---")

    # 侧边栏 - 参数设置
    st.sidebar.header("⚙️ 参数设置")

    st.sidebar.subheader("Cellpose参数")
    diameter = st.sidebar.slider("预期细胞直径 (像素)", 10, 100, 30, 5,
                                help="细胞的预期直径，None表示自动估计")
    flow_threshold = st.sidebar.slider("流场阈值", 0.0, 1.0, 0.4, 0.05,
                                      help="越高越严格，减少假阳性")
    cellprob_threshold = st.sidebar.slider("细胞概率阈值", -6.0, 6.0, 0.0, 0.5,
                                          help="细胞概率的阈值")

    st.sidebar.subheader("过滤参数")
    min_area = st.sidebar.slider("最小面积 (像素²)", 50, 200, 80, 10)
    max_area = st.sidebar.slider("最大面积 (像素²)", 500, 2000, 1000, 100)
    max_intensity = st.sidebar.slider("最大灰度值", 80, 180, 120, 10,
                                     help="过滤高灰度背景区域")
    min_circularity = st.sidebar.slider("最小圆度", 0.1, 0.5, 0.20, 0.05,
                                       help="圆度 = 4π×面积/周长²")

    # 文件上传
    uploaded_file = st.file_uploader("上传细胞图像", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

    if uploaded_file is not None:
        # 读取图像
        image = Image.open(uploaded_file)

        # 显示原图
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图像")
            st.image(image, use_container_width=True)

        # 执行分割按钮
        if st.button("🚀 开始分割", type="primary"):
            try:
                # 创建分割对象
                segmenter = CellSegmentationCellpose(image)

                # 预加载模型
                if segmenter.model is None:
                    segmenter.model = load_cellpose_model_cached('cyto')
                    if segmenter.model is None:
                        st.error("无法加载Cellpose模型")
                        return

                # 执行分割
                segmenter.segment_cells(
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold
                )

                # 计算属性
                segmenter.calculate_properties(
                    min_area=min_area,
                    max_area=max_area,
                    max_intensity=max_intensity,
                    min_circularity=min_circularity
                )

                # 保存到session state
                st.session_state['segmenter'] = segmenter
                st.session_state['segmented'] = True

            except Exception as e:
                st.error(f"分割过程出错: {e}")
                import traceback
                st.code(traceback.format_exc())

    # 显示分割结果
    if 'segmented' in st.session_state and st.session_state['segmented']:
        segmenter = st.session_state['segmenter']

        st.markdown("---")
        st.subheader("📊 分割结果")

        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("检测到的细胞数", len(segmenter.regions))
        with col2:
            areas = [r.area for r in segmenter.regions]
            st.metric("平均面积", f"{np.mean(areas):.1f} px²")
        with col3:
            circularities = []
            for r in segmenter.regions:
                if r.perimeter > 0:
                    circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                    circularities.append(circ)
            st.metric("平均圆度", f"{np.mean(circularities):.3f}")
        with col4:
            intensities = [getattr(r, 'intensity_mean', getattr(r, 'mean_intensity', 0))
                          for r in segmenter.regions]
            st.metric("平均灰度", f"{np.mean(intensities):.1f}")

        # 显示分割结果图像
        with col2:
            st.subheader("分割结果")
            result_image = segmenter.create_segmentation_image()
            st.image(result_image, use_container_width=True)

        # 细胞列表
        st.markdown("---")
        st.subheader("🔍 细胞详情")

        # 创建细胞选择器
        cell_options = [f"细胞 #{r.label}" for r in segmenter.regions]
        selected_cell = st.selectbox("选择细胞查看详情", cell_options)

        if selected_cell:
            # 获取选中的细胞
            cell_index = int(selected_cell.split('#')[1])
            region = None
            for r in segmenter.regions:
                if r.label == cell_index:
                    region = r
                    break

            if region:
                col1, col2 = st.columns([1, 2])

                with col1:
                    # 显示细胞放大图
                    st.subheader("细胞放大图")
                    cell_image = segmenter.create_cell_detail_image(region)
                    st.image(cell_image, use_container_width=True)

                with col2:
                    # 显示细胞参数
                    st.subheader("细胞参数")
                    params = segmenter.get_cell_parameters(region)

                    # 创建参数表格
                    param_data = {
                        "参数": list(params.keys()),
                        "数值": [f"{v:.2f}" if isinstance(v, float) else str(v)
                                for v in params.values()]
                    }
                    st.table(param_data)

                    # 显示位置信息
                    st.subheader("位置信息")
                    y0, x0 = region.centroid
                    minr, minc, maxr, maxc = region.bbox
                    st.write(f"- 中心坐标: ({x0:.1f}, {y0:.1f})")
                    st.write(f"- 边界框: ({minc}, {minr}) - ({maxc}, {maxr})")

    # 页脚
    st.markdown("---")
    st.markdown("""
    ### 关于Cellpose
    Cellpose是一个基于深度学习的通用细胞分割算法，使用预训练模型可以处理各种类型的细胞图像。

    **优势:**
    - 更准确的细胞边界检测
    - 更好地处理粘连细胞
    - 适应不同的细胞形态

    **参数说明:**
    - **预期细胞直径**: 细胞的大致直径，影响分割尺度
    - **流场阈值**: 控制分割的严格程度，越高越保守
    - **细胞概率阈值**: 判断是否为细胞的概率阈值
    """)


if __name__ == '__main__':
    main()
