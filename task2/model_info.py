import torch
from torchinfo import summary
from CNN import ResNet18
from VIT import ViT
# 假设你的模型定义在另一个文件或代码段中，这里导入它们
# from your_model_file import ResNet18, ViT

# 创建模型实例
cnn_model = ResNet18()
transformer_model = ViT()

# 使用torchinfo提供模型的详细摘要
# 假设输入数据大小为(batch_size, channels, height, width)
summary(cnn_model, input_size=(1, 3, 32, 32))
summary(transformer_model, input_size=(1, 3, 32, 32))
