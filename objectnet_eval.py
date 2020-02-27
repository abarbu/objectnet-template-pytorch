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
import json

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
parser.add_argument('--workers', default=multiprocessing.cpu_count(), type=int, metavar='N',
                    help='number of data loading workers (default: total num CPUs)')
parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int, metavar='N',
                    help='number of GPUs to use')
parser.add_argument('--batch_size', default=96, type=int, metavar='N',
                    help='mini-batch size (default: 96), this is the '
                         'batch size of each GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--softmax', default=True, type=bool, metavar='T/F',
                    help="apply a softmax function to network outputs to convert output magnitudes to confidence values (default:True)")
parser.add_argument('--convert_outputs_mode', default=1, type=int, metavar='N',
                    help="0: no conversion of prediction IDs, 1: convert from pytorch ImageNet prediction IDs to ObjectNet prediction IDs (default:1)")
args = parser.parse_args()

# check the Args
# images
assert (os.path.exists(args.images)), "Path to images folder: "+args.images+", does not exist!"
# output file
assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"
assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"
# model class name
try:
    getattr(model_description, args.model_class_name)
except AttributeError as e:
    print("Module: " + args.model_class_name + ", can not be found in model_description.py!")
    raise
# model check point file
assert (os.path.exists(args.model_checkpoint)), "Model checkpoint file: "+args.model_checkpoint+", does not exist!"
# workers
assert (args.workers <= multiprocessing.cpu_count()), "Number of workers: "+args.workers + ", should be <= the number of CPUs " + multiprocessing.cpu_count()+"!"
assert (args.workers >= 1), "Number of workers must be >= 1!"
# GPUs
assert (torch.cuda.is_available()), "No GPUs detected!"
assert (args.gpus <= torch.cuda.device_count()), "Requested "+args.gpus+" ,but only "+torch.cuda.device_count()+" are availible!"
assert (args.gpus >= 1), "You have to use at least 1 GPU!"
# batch batch_size
assert (args.batch_size >= 1), "Batch size must be >= 1!"
#convert outputs
assert (args.convert_outputs_mode in (0,1)), "Convert outputs mode must be either 0 or 1!"

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
device = torch.device("cuda:"+str(all_parallel_devices[0]))
batches_per_device = args.batch_size #upper bound estimate of how much data will fit in GPU memory, tune this based ou GPU memory availible


batch_size = (batches_per_device*len(all_parallel_devices))
num_workers = args.workers

img_format = "jpg"

mapping_file = "mapping_files/imagenet_pytorch_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)

def load_pretrained_net():
    print("initializing model ...")
    model = getattr(model_description, MODEL_CLASS_NAME)()

    print("loading pretrained weights from disk ...")
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, args.model_checkpoint)
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
            if args.softmax:
                sm = torch.nn.Softmax(dim=1)
            else:
                sm = lambda *args: args
            prediction_confidence, prediction_class = sm(model(inputs)).topk(5, 1)

        prediction_class = prediction_class.data.cpu().tolist()
        prediction_confidence = prediction_confidence.data.cpu().tolist()

        for i in range(len(fileName)):
            if args.convert_outputs_mode == 1:
                pytorchImageNetIDToObjectNetID(prediction_class[i])
            predictions.append([fileName[i]] + prediction_class[i] + prediction_confidence[i])
    return predictions


def pytorchImageNetIDToObjectNetID(prediction_class):
    for i in range(prediction_class):
        prediction_class[i] = mapping[prediction_class[i]]


objectnet_predictions = evalModels()
with open(args.output_file, 'w') as csvOut:
    csvwriter = csv.writer(csvOut, delimiter=',')
    for predictImg in objectnet_predictions:
        csvwriter.writerow(predictImg)
