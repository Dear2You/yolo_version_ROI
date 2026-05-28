# YOLOv8 可视化目标检测系统

这是一个基于 PyQt5、OpenCV 和 Ultralytics YOLO 的本地目标检测桌面应用。项目提供模型加载、图片/文件夹/视频/摄像头/远程视频流检测、结果可视化、检测日志导出、异常报警、实验分析与报告生成等功能，适合用于 YOLO 模型演示、检测实验记录和参数对比。

## 主要功能

- 模型加载：支持 `.pt`、`.onnx`、`.engine`、`.torchscript` 等 YOLO 模型文件。
- 多源检测：支持单张图片、图片文件夹、本地视频、摄像头和 RTSP/HTTP 远程视频流。
- 推理参数调节：可设置置信度、IoU、输入尺寸，并控制是否显示类别标签和置信度。
- 结果展示：显示检测图像、目标数量、FPS、检测摘要和类别结果表。
- 原图对比：可切换原图与检测结果的并排对比视图。
- 检测热力图：根据检测框位置生成热力覆盖图，便于观察目标分布。
- ROI 与异常报警：支持重点区域 ROI、低置信度、目标数量突变、连续无目标、类别数量超限等规则报警。
- 自动截图：出现异常/报警时，可自动保存异常帧图片和对应 JSON 元数据。
- 结果保存：可保存图片、文件夹检测结果、视频检测结果和摄像头录制结果。
- 日志导出：支持 CSV、JSON、YOLO TXT 标注目录和 COCO JSON 标注格式。
- 实验分析中心：提供历史记录、当前统计、统计图表、输入尺寸对比、类别阈值、任务队列、历史检索和参数预设。
- 报告生成：可导出 Markdown 实验报告和 Word 实验报告。
- 本地后端：使用 SQLite 记录检测会话、帧级数据和检测结果。
- 资源监控：展示 CPU、内存、GPU 和显存状态。
- 主题切换：支持浅色/深色界面主题。

## 项目结构

```text
yolov8_version111/
├── main.py                         # 应用入口
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
└── app/
    ├── config.py                   # 全局配置
    ├── backend/
    │   └── local_backend.py         # SQLite 本地数据后端
    ├── services/
    │   └── detector_service.py      # YOLO 模型加载、推理与结果解析
    ├── ui/
    │   ├── main_window.py           # 主窗口、交互逻辑、分析中心
    │   └── styles.py                # 界面主题与样式
    ├── utils/
    │   ├── file_utils.py            # 文件与目录工具
    │   └── image_utils.py           # OpenCV 图像转 Qt Pixmap
    └── workers/
        ├── detection_worker.py      # 文件夹/视频/摄像头检测线程
        └── benchmark_worker.py      # 输入尺寸性能对比线程
```

运行后会在项目根目录下生成 `_runtime/`，用于保存 SQLite 数据库和异常截图等运行数据。

## 环境要求

- Python 3.10 或更高版本
- Windows 环境下测试更合适，项目界面基于 PyQt5
- NVIDIA GPU + CUDA 环境可获得更好的推理速度
- 若没有 CUDA，也可以使用 CPU 推理，但速度会明显下降

依赖文件默认使用 CUDA 12.8 版本 PyTorch：

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
onnxruntime-gpu==1.23.2
tensorrt-cu12==10.16.1.11
```

如果本机 CUDA 版本不匹配，请根据自己的环境调整 `requirements.txt` 中的 PyTorch、ONNX Runtime 或 TensorRT 版本。

## 安装与运行

1. 创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. 安装依赖：

```powershell
pip install -r requirements.txt
```

3. 启动程序：

```powershell
python main.py
```

## 基本使用流程

1. 点击“模型载入”，选择 YOLO 模型文件。
2. 选择检测来源：
   - “图片检测”：检测单张图片。
   - “文件夹检测”：批量检测文件夹内图片。
   - “视频检测”：检测本地视频文件。
   - “摄像头检测”：调用本机摄像头实时检测。
   - “远程流”：输入 RTSP 或 HTTP 连续视频流地址。
3. 在参数区域调整置信度、IoU、输入尺寸和显示选项。
4. 点击“开始检测”执行任务。
5. 检测完成后，可点击“保存结果”保存可视化结果。
6. 点击“导出日志”导出检测记录或标注文件。
7. 点击“实验分析”查看历史记录、统计图表、参数对比和报告导出。

## 支持的数据格式

图片输入：

```text
.jpg .jpeg .png .bmp .tif .tiff .webp
```

视频输入：

```text
.mp4 .avi .mov .mkv .flv .wmv
```

模型输入：

```text
.pt .onnx .engine .torchscript
```

日志与标注导出：

```text
CSV 检测日志
JSON 检测日志
YOLO TXT 标注目录
COCO JSON 标注文件
```

实验报告导出：

```text
Markdown 报告
Word DOCX 报告
```

## 实验分析中心

“实验分析”窗口包含以下功能：

- 实验历史：查看最近检测会话、模型、后端、输入尺寸、状态、采样帧、目标数和平均 FPS。
- 当前统计：统计当前会话的帧数、目标数、类别分布、FPS 和置信度。
- 统计图表：生成类别数量分布图、FPS 与目标数量帧级趋势图，并可导出 PNG。
- 参数对比：自动测试 320、416、640 等输入尺寸下的速度和检测数量。
- 类别阈值：为指定类别设置独立最低置信度，过滤展示、日志和导出结果。
- 任务队列：批量加入图片、文件夹、视频或远程流任务并顺序执行。
- 历史检索：按类别、来源、模型或状态关键词检索历史检测结果。
- 参数预设：保存和复用当前检测参数、ROI 和报警设置。

## 运行数据说明

程序会在项目根目录下自动创建 `_runtime/`：

```text
_runtime/
├── detection_backend.sqlite3        # 检测会话、帧记录和检测结果
└── alert_captures/                  # 异常帧截图和 JSON 元数据
```

临时检测结果会写入系统临时目录中的 `yolo_visual_project/`，保存结果时再复制到用户选择的位置。

## 注意事项

- 运行检测前必须先加载模型。
- 远程流需要填写连续视频流地址，例如 `rtsp://...` 或 `http://.../video`，不要填写监控网页地址或单张截图地址。
- 文件夹全量保存会为每张图片绘制并写入结果图，速度会低于只显示预览的模式。
- TensorRT `.engine` 模型通常与生成它的 GPU、TensorRT 和 CUDA 环境强相关，迁移到其他机器可能需要重新导出。
- 如果安装 GPU 依赖失败，请先确认 CUDA、显卡驱动、Python 版本和 PyTorch 轮子版本是否匹配。
