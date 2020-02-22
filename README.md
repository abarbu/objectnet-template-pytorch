# ObjectNet competition eval model example code
Baseline model for the ObjectNet competition

Uses the model from https://github.com/facebookresearch/WSL-Images

# Requirements
python 3\
tqdm\
pytorch 1.4\
cuda 10.1

# Instructions to run objectnet_eval.py
cd model\
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth \
cd ../ \
python objectnet_eval.py images_folder out.csv resnext101_32x48d_wsl model/ig_resnext101_32x48-3e41cc8a.pth

# objectnet_eval.py --help
```
usage: objectnet_eval.py [-h] [--workers N] [--gpus GPUS] [--batch_size N]
                         images-dir output-file model-class-name
                         model-checkpoint

Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV
file.

positional arguments:
  images-dir        path to dataset
  output-file       path to predictions output file
  model-class-name  model class name in model_description.py
  model-checkpoint  path to model checkpoint

optional arguments:
  -h, --help        show this help message and exit
  --workers N       number of data loading workers (default: total num CPUs)
  --gpus GPUS       number of GPUs to use
  --batch_size N    mini-batch size (default: 256), this is the batch size of
                    each GPU on the current node when using Data Parallel or
                    Distributed Data Parallel
```

# Code structure
objectnet_eval.py
- loads the pretrained model
- evaluates batches of images using parallel dataloading and 1 or more GPUs
- aggregates the predictions and writes them to a CSV

objectnet_pytorch_dataloader.py
- extends pytorch VisionDataset class, allows ObjectNetDataset to be passed to pytorch DataLoader for parallel data loading
- scans all the images in the input images folder and makes a list of files, ignores subdirectory folder structures
- loads images using pil image loader
- applies transforms specified in data_transformation_description.py and crops out 2 pixel red border on ObjectNet images

Inside of the model directory: (*This is the only code that most competitors will have to modify) \
model_description.py
- pytorch model description class that can be built out of torchvision module blocks or can extend nn.Module to implement any neural net model
- pretrained weights for the model parameters are loaded in objectnet_eval.py -> load_pretrained_net() by model.load_state_dict
- the current example model is a resnext101_32x48d

data_transform_description.py
- contains all the dataset preprocessing transformations except cropping out the 2px red pixel border
- getTransform() returns a pytorch transform composition
