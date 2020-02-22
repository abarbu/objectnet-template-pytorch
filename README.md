# objectnet_competition_demo
baseline mode for the ObjectNet competition

Uses the model from https://github.com/facebookresearch/WSL-Images

# Instructions to run it
cd model
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth
cd ../
python objectnet_eval.py images_folder/ out.csv resnext101_32x48d_wsl model/ig_resnext101_32x48-3e41cc8a.pth
