
This repository contains instructions on how to build a docker image using the PyTorch deep learning framework for the ObjectNet Challenge (**#####SH Proper name for challenge and link to synapse wiki**). It assumes you already have a pre-trained PyTorch model which you intend to submit for evaluation to the ObjectNet Challenge.

If your model is built using a different framework the docker template provided will require additional customisation, instructions for which are provided on the ObjectNet Challenge wiki page **#####SH Add link to wiki**

If you are not familiar with docker here are instructions on how to [install docker](https://docs.docker.com/install/), along with a [quick start guide](https://docs.docker.com/get-started/).

These instructions are split into two sections:
 - *Section 1* which describes how to:
   1. run the example code & model on a local machine, and
   2. plug in your own model into this example and test on a local machine.
 - *Section 2* which describes how to create a docker image ready to submit to the challenge.

# Section 1: ObjectNet competition eval model example code
The following section provides example code and a
baseline [model](https://github.com/facebookresearch/WSL-Images) for the ObjectNet competition (**#####SH Proper name for challenge**).
The code is structured such that most existing PyTorch models can
be plugged into the example with minimal code changes necessary.

The example code uses batching and parallel data loading to improve inference
efficiency.

**Note:** If you are building your own customized docker image with your own
code it is highly recommended to use similar optimized techniques to ensure
your submission will complete within the time limit set by the challenge organisers.


## 1.1 Requirements
The following libraries are required to run this example and must be installed
on the local test machine. The same libraries will be automatically installed
into the Docker image when the image is built.
 - python 3
 - tqdm
 - pytorch 1.4
 - cuda 10.1
**#####SH test on non gpu machine w/o cuda**

## 1.2 Install NVIDIA drivers
If your local machine has NVIDIA-capable GPUs and you want to test your docker image  
locally using these GPUs then you will need to ensure the NVIDIA drivers have been
installed on your test machine. See
.
Instructions on how to install the CUDA toolkit and NVIDIA drivers can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation). Be sure to match the versions of CUDA/NVIDIA installed with the version of PyTorch and CUDA used to build your docker image (see [Section 2: Building the docker image](Section_2:_Building_the_docker_image)).

## 1.3 Clone this repository
Clone this repo to a machine which has docker installed:
```bash
git clone https://github.com/dmayo/objectnet_competition_demo.git
```

## 1.4 objectnet_eval.py
`objectnet_eval.py` is the main entry point for running this example.
Full help is available using `objectnet_eval.py --help`:
```
usage: objectnet_eval.py [-h] [--workers N] [--gpus N] [--batch_size N]
                         [--softmax T/F] [--convert_outputs_mode N]
                         images-dir output-file model-class-name
                         model-checkpoint

Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV
file.

positional arguments:
  images-dir            path to dataset
  output-file           path to predictions output file
  model-class-name      model class name in model_description.py
  model-checkpoint      path to model checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --workers N           number of data loading workers (default: total num
                        CPUs)
  --gpus N              number of GPUs to use
  --batch_size N        mini-batch size (default: 96), this is the batch size
                        of each GPU on the current node when using Data
                        Parallel or Distributed Data Parallel
  --softmax T/F         apply a softmax function to network outputs to convert
                        output magnitudes to confidence values (default:True)
  --convert_outputs_mode N
                        0: no conversion of prediction IDs, 1: convert from
                        pytorch ImageNet prediction IDs to ObjectNet
                        prediction IDs (default:1)
```

## 1.5 Code structure
There follows a description of the code structure used for this example.

*./objectnet_eval.py:*
- loads the pre-trained model (defined in model-class-name & model-checkpoint file)
- pre-trained weights for the model parameters are loaded in objectnet_eval.py -> load_pretrained_net() by model.load_state_dict
- evaluates batches of images using parallel dataloading and 1 or more GPUs (--gpus)
- aggregates the predictions and writes them to a CSV file (output-file)

*./objectnet_pytorch_dataloader.py:*
- extends pytorch VisionDataset class, allows ObjectNetDataset to be passed to pytorch DataLoader for parallel data loading
- scans all the images in the input images folder and makes a list of files, ignores subdirectory folder structures
- loads images using pil image loader
- applies transforms specified in data_transform_description.py and crops out 2 pixel red border on ObjectNet images

Inside of the model directory: (*This is the only code that you will have to modify*):

*./model/model_description.py:*
- pytorch model description class that can be built out of torchvision module blocks or can extend nn.Module to implement any neural net model
- the current example model is a [resnext101_32x48d](https://github.com/facebookresearch/WSL-Images)
- add your own model description class to this file. **#####SH do we need to explain what the numbers are**

*./model/data_transform_description.py:*
- contains all the dataset preprocessing transformations except cropping out the 2px red pixel border
- getTransform() returns a pytorch transform composition
- include any customized data transformation code your model requires in this class.

## 1.6 Testing the example
Before executing the example
for the first time you must download the sample model as shown below.
```bash
# Download the model:
cd model
wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth
```
You must also place a set of test images into the folder specified by the `images-dir`
argument ('/input' in the example below) to the `objectnet_eval.py` program.

Then run `objectnet_eval.py` using the following arguments:
```bash
# Perform batch inference:
python objectnet_eval.py /input out.csv resnext101_32x48d_wsl model/ig_resnext101_32x48-3e41cc8a.pth
```
Results will be written to the `out.csv` file in the current directory. Check
the output conforms to the format expected by the ObjectNet Challenge **##### link to
format description**

## 1.7 Testing your own model with the example
To test the example with your own model:
1. Copy your model checkpoint file into the `./model` directory.
2. Add your model description as a class to `./mode/model_description.py`. The
class name will be used as the `model-class-name` argument to `objectnet_eval.py`.
For example, for a model which has 32 groups and width per-group of 16 we could add:
```python
class my_model(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=16)
```
3. Add any custom image transformation code your model requires to the `./model/data_transform_description.py` module.
4. Test your model's inference using:
```bash
python objectnet_eval.py /input out.csv my_model model/my_model.pth
```
5. Ensure the output is in the format expected by the ObjectNet Challenge
**##### link to format description**



---


# Section 2: Building the docker image

## 2.1 Install NVIDIA drivers
Prior to uploading the docker image to the competition portal for evaluation you should test your docker image locally. If your local machine has NVIDIA-capable GPUs and you wish to test inference using GPUs then you will first need to install the NVIDIA drivers on your machine. See
section [1.2 Install NVIDIA drivers](1.2_Install_NVIDIA_drivers) above.

## 2.2 Add your model & supporting code
Ensure you have been able to successfully test your model on the local host using the `objectnet_eval.py` example code - see section [1.7 Testing your own model with the example](1.7_Testing_your_own_model_with_the_example) for more details.

**#####SH is the below his true**

**Note:** Your model must have been saved using `torch.save(model, "<PATH TO SAVED MODEL FILE>")`.

## 2.3 Build the docker image
Docker images are built from a series of statements contained in a `Dockerfile`. A template Dockerfile is provided for models built using the PyTorch (`Dockerfile`) deep learning framework and saved using the `torch.save` api.

The PyTorch docker image for the ObjectNet Challenge uses the [official PyTorch docker images](https://hub.docker.com/r/pytorch/pytorch/tags) as its base image. These PyTorch images come with built-in GPU support and with python 3 pre-loaded. By default, the docker image is built using [PyTorch version 1.4, cuda 10.1 and cudann7](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-runtime/images/sha256-ee783a4c0fccc7317c150450e84579544e171dd01a3f76cf2711262aced85bf7?context=explore).

**#####SH - pointer to a tutorial on how to use the images**

 You can customise the PyTorch and cuda versions used for the base image by editing the `Dockerfile` file and uncommenting one of the following lines - choose the one which most closely matches the versions used to build your model:
   ```
   # You can select from the following pre-built PyTorch Docker images.
   # Select the image which most closely matches the
   # versions used to build your model.
   FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
   #FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
   #FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime
   #FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
   #FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
   ```

To improve performance the example code batches up inferencing of the ObjectNet images and execute a number of streams (or workers) in parallel.

**#####SH this might change**
You can further customise the build of you docker container by specifying the following arguments at build time:
 - MODEL_CLASS_NAME: The name of the model class you have defined in the `model_descriptions.py` file. This is passed as the `model-class-name` argument to the `objectnet_eval.py` module. For example "my_model".
 - MODEL_CHECKPOINT: The name of your pre-trained model file (the one you copied into the model directory). This value is passed as the `model-checkpoint` argument to the `objectnet_eval.py` module. For example "my_model.pth".
 - WORKERS: The number of parallel workers you would like to use. The rule of thumb is one worker per CPU core on your machine
 - BATCH_SIZE: The number of images per batch. 16 is a good starting point.

**#####SH we probably don't want to publicise workers & batch size since this could in theory lead to a DoS attack**

**Note:** The ObjectNet Challenge submission process is expecting the output to be directed to `\output\predictions.csv` file within the container image. Ensure the `output-file` argument to the `objectnet_eval.py` module of the ENTRYPOINT command in the `Dockerfile` file is set to `\output\predictions.csv`. For example:
```
# Define the command to execute when the container is run
ENTRYPOINT python objectnet_eval.py /input /output/predictions.csv $MODEL_CLASS_NAME $MODEL_PATH
```

To build the docker image run:
```bash
docker build --build-arg MODEL_CLASS_NAME="resnext101_32x48d_wsl" --build-arg MODEL_CHECKPOINT="ig_resnext101_32x48-3e41cc8a.pth" -t docker.synapse.org/<Your Synapse Project ID>/<Repo name>:<Tag> -f Dockerfile .
```
Replace \<Your Synapse Project ID> with the [ID of the Synapse project](https://docs.synapse.org/articles/getting_started.html#synapse-ids) you have registered with the Challenge.
For example:
```bash
# With version tagging:
docker build --build-arg MODEL_CLASS_NAME="resnext101_32x48d_wsl" --build-arg MODEL_CHECKPOINT="ig_resnext101_32x48-3e41cc8a.pth" -t docker.synapse.org/syn12345/my-model:version1 -f Dockerfile .
# Or without version tagging:
docker build --build-arg MODEL_CLASS_NAME="resnext101_32x48d_wsl" --build-arg MODEL_CHECKPOINT="ig_resnext101_32x48-3e41cc8a.pth" -t docker.synapse.org/syn12345/my-model -f Dockerfile .
```
Once the build is complete your newly built docker image can be listed using  the command:
```bash
$ docker images
```

## 2.4 Testing the docker image locally
Test the docker image locally before submitting it to the challenge. For example, if you tagged your docker image during the build step with `docker.synapse.org/syn12345/my-model:version1`, then from the root directory of this cloned repo issue:

**#####SH ONLY works with GPUs at the moment - see below example**
```bash
docker run -ti --rm -v $PWD/sample-images:/input/ -v $PWD/output:/output docker.synapse.org/syn12345/my-model:version1
```
The `-v $PWD/sample-images:/input` mounts a directory of test images from the local path into `/input` within the docker container. Similarly, `-v $PWD/output:/output` mounts an output directory from the local path into `/output` of the container.

If your test host has GPUs and you built a GPU enabled docker image, then add the `--gpus=all` parameter to the `docker run` command in order to utilise your GPUs:
```bash
docker run -ti --rm --gpus=all -v $PWD/sample-images:/input/ -v $PWD/output:/output docker.synapse.org/syn12345/my-model:version1
```

## 2.5 Debugging your docker image locally
If there are errors during Step 5 then you will need to debug your docker container.
If you make changes to your code there is no need to rebuild the docker container. To quickly test your new code, simply mount the root path of this repo as a volume when you run the container. For example:
```bash
docker run -ti --rm -v $PWD:/workspace -v $PWD/sample-images:/input/ -v $PWD/output:/output docker.synapse.org/syn12345/my-model:version1
```
When the docker container is run, the local `$PWD` will be mounted over `/workspace` directory within the image which effectively means any code/model changes made since the last `docker build` command will be contained within the running container.


---

# Upload your docker image to Synapse:
Once you have built and tested your docker image locally you should upload it to the [Synapse docker registry](https://www.synapse.org/#!Synapse:syn21445381/wiki/600093) and then [submit your docker image to the challenge](https://www.synapse.org/#!Synapse:syn21445381/wiki/600093). **#####SH Will need to update both these link**
