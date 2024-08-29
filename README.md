# Histogram Equalization
## 如何运行

1. Python3.9.6，以及需要的lib
2. 修改`main.py`中的`Flag`以选择要绘制的图形。
3. 运行`main.py`以测试结果。

## 项目结构

- `main.py`：主测试脚本，用于运行和测试算法。
- `basic_he.py`：包含基本的直方图均衡化算法实现。
- `utils.py`：包含各种辅助函数，主要用于数据可视化。
- `pics`：存放图像样本的文件夹。
- `results`: 存放输出图像
- `wgif`: 使用wgif增强图像的实现

## Flag功能

- `DRAW_HISTOGRAM`：绘制直方图。
- `DRAW_PICS`：绘制原始和处理后的图片。
- `DRAW_3D_MAP`：绘制3D可视化的灰度矩阵。
- `METHOD`：选择方法
- `RESTORT_COLOR`: 输出彩色图像
