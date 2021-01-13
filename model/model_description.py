from torchvision.models.resnet import ResNet, Bottleneck

class resnext101_32x48d_wsl(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=48)

class CustomModel(ResNet):
    def __init__(self):
        #'wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
        super().__init__(Bottleneck, [3, 4, 6, 3], width_per_group= 64 * 2)