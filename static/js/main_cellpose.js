// 全局变量
let currentImage = null;
let imageWidth = 0;
let imageHeight = 0;
let canvas = null;
let ctx = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');

    // 上传按钮点击事件
    document.getElementById('uploadBtn').addEventListener('click', function() {
        document.getElementById('imageInput').click();
    });

    // 文件选择事件
    document.getElementById('imageInput').addEventListener('change', handleFileSelect);

    // Canvas点击事件
    canvas.addEventListener('click', handleCanvasClick);
});

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // 显示文件名
    document.getElementById('fileName').textContent = file.name;

    // 显示加载指示器
    document.getElementById('loadingIndicator').style.display = 'inline-flex';

    // 获取参数
    const diameter = parseInt(document.getElementById('diameter').value);
    const flowThreshold = parseFloat(document.getElementById('flowThreshold').value);
    const cellprobThreshold = parseFloat(document.getElementById('cellprobThreshold').value);
    const minArea = parseInt(document.getElementById('minArea').value);
    const maxArea = parseInt(document.getElementById('maxArea').value);
    const maxIntensity = parseInt(document.getElementById('maxIntensity').value);
    const minCircularity = parseFloat(document.getElementById('minCircularity').value);

    // 创建FormData
    const formData = new FormData();
    formData.append('image', file);
    formData.append('diameter', diameter);
    formData.append('flow_threshold', flowThreshold);
    formData.append('cellprob_threshold', cellprobThreshold);
    formData.append('min_area', minArea);
    formData.append('max_area', maxArea);
    formData.append('max_intensity', maxIntensity);
    formData.append('min_circularity', minCircularity);

    // 上传图像并执行分割
    fetch('/segment', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏加载指示器
        document.getElementById('loadingIndicator').style.display = 'none';

        if (data.error) {
            alert('错误: ' + data.error);
            return;
        }

        // 显示分割结果
        displaySegmentedImage(data);
    })
    .catch(error => {
        document.getElementById('loadingIndicator').style.display = 'none';
        alert('处理失败: ' + error);
    });
}

// 显示分割后的图像
function displaySegmentedImage(data) {
    currentImage = 'data:image/png;base64,' + data.result_image;

    // 加载图像
    const img = new Image();
    img.onload = function() {
        imageWidth = img.width;
        imageHeight = img.height;

        // 设置canvas尺寸
        const container = document.querySelector('.image-container');
        const maxWidth = container.clientWidth - 40;
        const maxHeight = container.clientHeight - 40;

        let scale = Math.min(maxWidth / imageWidth, maxHeight / imageHeight, 1);

        canvas.width = imageWidth * scale;
        canvas.height = imageHeight * scale;

        // 绘制图像
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // 显示canvas，隐藏placeholder
        document.querySelector('.placeholder').style.display = 'none';
        canvas.style.display = 'block';

        // 显示图像信息
        document.getElementById('imageInfo').style.display = 'flex';
        document.getElementById('cellCount').textContent = `检测到 ${data.stats.cell_count} 个细胞 (平均面积: ${data.stats.avg_area.toFixed(1)}px², 平均圆度: ${data.stats.avg_circularity.toFixed(3)})`;
        document.getElementById('imageSize').textContent = `图像尺寸: ${imageWidth} × ${imageHeight}`;
    };
    img.src = currentImage;
}

// 处理Canvas点击
function handleCanvasClick(event) {
    if (!currentImage) return;

    // 获取点击位置
    const rect = canvas.getBoundingClientRect();
    const scaleX = imageWidth / canvas.width;
    const scaleY = imageHeight / canvas.height;

    const x = Math.floor((event.clientX - rect.left) * scaleX);
    const y = Math.floor((event.clientY - rect.top) * scaleY);

    // 请求细胞信息
    fetch('/get_cell_info', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ x: x, y: y })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error || !data.success) {
            console.log('未点击到细胞');
            return;
        }

        // 显示细胞信息
        displayCellInfo(data.cell);
    })
    .catch(error => {
        console.error('获取细胞信息失败:', error);
    });
}

// 显示细胞信息
function displayCellInfo(cell) {
    // 隐藏placeholder，显示内容
    document.getElementById('cellInfoPlaceholder').style.display = 'none';
    document.getElementById('cellInfoContent').style.display = 'block';

    // 设置细胞编号
    document.getElementById('cellLabel').textContent = cell.label;

    // 设置细胞放大图
    document.getElementById('cellImage').src = 'data:image/png;base64,' + cell.cell_image;

    // 设置参数
    document.getElementById('majorAxis').textContent = cell.major_axis.toFixed(2);
    document.getElementById('minorAxis').textContent = cell.minor_axis.toFixed(2);
    document.getElementById('circularity').textContent = cell.circularity.toFixed(3);
    document.getElementById('area').textContent = cell.area.toFixed(0);
    document.getElementById('perimeter').textContent = cell.perimeter.toFixed(2);
    document.getElementById('meanIntensity').textContent = cell.mean_intensity.toFixed(2);
    document.getElementById('variance').textContent = cell.variance.toFixed(2);

    // 更新圆度指示器
    const circularityPercent = cell.circularity * 100;
    document.getElementById('circularityBar').style.width = circularityPercent + '%';

    // 根据圆度设置颜色
    if (cell.circularity < 0.5) {
        document.getElementById('circularityBar').style.backgroundColor = '#e74c3c';
    } else if (cell.circularity < 0.7) {
        document.getElementById('circularityBar').style.backgroundColor = '#f39c12';
    } else {
        document.getElementById('circularityBar').style.backgroundColor = '#27ae60';
    }
}
