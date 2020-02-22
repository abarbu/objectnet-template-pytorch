# objectnet_competition_demo
baseline model for the ObjectNet competition

Uses the model from https://github.com/facebookresearch/WSL-Images

# Requirements
python 3
pytorch 1.4
tqdm

# Instructions to run objectnet_eval.py
cd model\
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth \
cd ../ \
python objectnet_eval.py images_folder out.csv resnext101_32x48d_wsl model/ig_resnext101_32x48-3e41cc8a.pth

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

inside the model directory: (This is the only code that most competitors will have to modify \
model_description.py
- pytorch model description class that can be built out of torchvision module blocks or can extend nn.Module to implement any neural net model
- pretrained weights for the model parameters are loaded in objectnet_eval.py -> load_pretrained_net() by model.load_state_dict
- the current example model is a resnext101_32x48d

data_transform_description.py
- contains all the dataset preprocessing transformations except cropping out the 2px red pixel border
- getTransform() returns a pytorch transform composition
