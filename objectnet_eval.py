import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm as tqdm
import multiprocessing
import os
import sys
import argparse
import csv

# import the PyTorch wrapper for ObjectNet
from objectnet_pytorch_dataloader import ObjectNetDataset as ObjectNet

sys.path.insert(0, 'model')
import model_description
from data_transform_description import data_transform

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV file.')
parser.add_argument('images', metavar='images-dir',
                    help='path to dataset')
parser.add_argument('output_file', metavar='output-file',
                    help='path to predictions output file')
parser.add_argument('model_class_name', metavar='model-class-name',
                    help='model class name in model_description.py')
parser.add_argument('model_checkpoint', metavar='model-checkpoint',
                    help='path to model checkpoint')
parser.add_argument('-j', '--workers', default=multiprocessing.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: total num CPUs)')
parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int,
                    help='number of GPUs to use')
parser.add_argument('-b', '--batch_size', default=96, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the '
                         'batch size of each GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

args = parser.parse_args()

#input images path
print()
print("**** params ****")
for k in vars(args):
    print(k,vars(args)[k])
OBJECTNET_IMAGES_FOLDER = args.images
print("****************")
print()

MODEL_CLASS_NAME = args.model_class_name

# model is copied to all parallel devies and each device evaluates a portion of a batch
all_parallel_devices = [i for i in range(args.gpus)] #list of GPU IDs to use for model evaluation
device = torch.device("cuda:"+str(all_parallel_devices[0]) if torch.cuda.is_available() else "cpu")
batches_per_device = args.batch_size #upper bound estimate of how much data will fit in GPU memory, tune this based ou GPU memory availible


batch_size = (batches_per_device*len(all_parallel_devices))
num_workers = args.workers

img_format = "jpg"

def load_pretrained_net():
    print("initializing model ...")
    model = getattr(model_description, MODEL_CLASS_NAME)()

    print("loading pretrained weights from disk ...")
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, "model/ig_resnext101_32x48-3e41cc8a.pth")
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    trans = data_transform().getTransform()

    return {'model': model, 'transform': trans}


def evalModels():
    '''
    returns:  [
                imageFileName1, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                imageFileName2, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                ...
              ]
    '''

    net = load_pretrained_net()

    objectnet = ObjectNet(OBJECTNET_IMAGES_FOLDER,transform=net['transform'], img_format=img_format)
    dataloader = DataLoader(objectnet, batch_size=(batches_per_device*len(all_parallel_devices)), num_workers=num_workers*2, pin_memory=True)

    predictions=[]
    for inputs, fileName in tqdm(dataloader, desc="Evaluating "+MODEL_CLASS_NAME+" ..."):
        with torch.no_grad():
            model=nn.DataParallel(net['model'], device_ids=all_parallel_devices)
            model.eval()
            model.to(device)
            inputs.to(device)
            sm = torch.nn.Softmax(dim=1)
            prediction_confidence, prediction_class = sm(model(inputs)).topk(5, 1)

        prediction_class = prediction_class.data.cpu().tolist()
        prediction_confidence = prediction_confidence.data.cpu().tolist()
        for i in range(len(fileName)):
            predictions.append([fileName[i]] + prediction_class[i] + prediction_confidence[i])
    return predictions

objectnet_predictions = evalModels()
with open(args.output_file, 'w') as csvOut:
    csvwriter = csv.writer(csvOut, delimiter=',')
    for predictImg in objectnet_predictions:
        csvwriter.writerow(predictImg)
