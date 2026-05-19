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

    // 创建FormData
    const formData = new FormData();
    formData.append('image', file);

    // 上传图像
    fetch('/upload', {
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
        alert('上传失败: ' + error);
    });
}

// 显示分割后的图像
function displaySegmentedImage(data) {
    currentImage = 'data:image/png;base64,' + data.image;
    imageWidth = data.width;
    imageHeight = data.height;

    // 加载图像
    const img = new Image();
    img.onload = function() {
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
        document.getElementById('cellCount').textContent = `检测到 ${data.cell_count} 个细胞`;
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
        if (data.error) {
            console.log('未点击到细胞');
            return;
        }

        // 显示细胞信息
        displayCellInfo(data);
    })
    .catch(error => {
        console.error('获取细胞信息失败:', error);
    });
}

// 显示细胞信息
function displayCellInfo(data) {
    // 隐藏placeholder，显示内容
    document.getElementById('cellInfoPlaceholder').style.display = 'none';
    document.getElementById('cellInfoContent').style.display = 'block';

    // 设置细胞编号
    document.getElementById('cellLabel').textContent = data.label;

    // 设置细胞放大图
    document.getElementById('cellImage').src = 'data:image/png;base64,' + data.cell_image;

    // 设置参数
    document.getElementById('majorAxis').textContent = data.major_axis.toFixed(2);
    document.getElementById('minorAxis').textContent = data.minor_axis.toFixed(2);
    document.getElementById('circularity').textContent = data.circularity.toFixed(3);
    document.getElementById('area').textContent = data.area.toFixed(0);
    document.getElementById('perimeter').textContent = data.perimeter.toFixed(2);
    document.getElementById('meanIntensity').textContent = data.mean_intensity.toFixed(2);
    document.getElementById('variance').textContent = data.variance.toFixed(2);

    // 更新圆度指示器
    const circularityPercent = data.circularity * 100;
    document.getElementById('circularityBar').style.width = circularityPercent + '%';

    // 添加高亮效果
    highlightCell(data.bbox);

    // 滚动到顶部
    document.querySelector('.info-panel').scrollTop = 0;
}

// 高亮选中的细胞
function highlightCell(bbox) {
    // 重新绘制图像
    const img = new Image();
    img.onload = function() {
        const scale = canvas.width / imageWidth;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // 绘制高亮框
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.strokeRect(
            bbox[0] * scale,
            bbox[1] * scale,
            (bbox[2] - bbox[0]) * scale,
            (bbox[3] - bbox[1]) * scale
        );
    };
    img.src = currentImage;
}
