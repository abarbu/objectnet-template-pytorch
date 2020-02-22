import torchvision.transforms as transforms

class data_transform:
    def __init__(self):
        self.model_pretrain_params = {}
        self.model_pretrain_params['input_size'] = [3, 224, 224]
        self.model_pretrain_params['mean'] = [0.485, 0.456, 0.406]
        self.model_pretrain_params['std'] = [0.229, 0.224, 0.225]
        self.resize_dim = self.model_pretrain_params['input_size'][1]

    def getTransform(self):
        trans = transforms.Compose([transforms.Resize(self.resize_dim),
                                    transforms.CenterCrop(self.model_pretrain_params['input_size'][1:3]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.model_pretrain_params['mean'],
                                                         std=self.model_pretrain_params['std'])
                                    ])
        return trans
