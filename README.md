# Important!
As this branch does not contain gpus, ensure the docker container is run with the following (i.e. no gpu tag):
```bash
docker run -ti --rm -v $PWD/input/images:/input/ -v $PWD/output:/output my-model:version1
```

# Overview
This repository contains instructions on how to build a docker image using the PyTorch deep learning framework for the [ObjectNet Challenge](https://www.synapse.org/#!Synapse:syn21445379/wiki/)**###AS Update to EvalAI**. It assumes you already have a pre-trained PyTorch model which you intend to submit for evaluation to the ObjectNet Challenge.

If your model is built using a different framework the docker template provided will require additional customisation, instructions for which are provided in section B of [Docker Image Creation](https://www.synapse.org/#!Synapse:syn21445379/wiki/601552)**###AS Update to EvalAI**.

If you are not familiar with docker here are instructions on how to [install docker](https://docs.docker.com/install/), along with a [quick start guide](https://docs.docker.com/get-started/).

These instructions are split into two sections:
 - *Section 1* which describes how to:
   1. run the example code & model on a local machine, and
   2. plug in your own model into this example and test on a local machine.
 - *Section 2* which describes how to create a docker image ready to submit to the challenge.

# Section 1: ObjectNet competition eval model example code
The following section provides example code and a
baseline [model](https://github.com/facebookresearch/WSL-Images) for the ObjectNet Challenge.
The code is structured such that most existing PyTorch models can
be plugged into the example with minimal code changes necessary.

The example code uses batching and parallel data loading to improve inference
efficiency.

**Note:** If you are building your own customized docker image with your own
code it is highly recommended to use similar optimized inferencing techniques to ensure
your submission will complete within the time limit set by the challenge organisers.


## 1.1 Requirements
The following libraries are required to run this example and must be installed
on the local test machine. The same libraries will be automatically installed
into the Docker image when the image is built.
 - python 3
 - tqdm
 - pytorch 1.4
 - cuda 10.1

## 1.2 Install NVIDIA drivers
If your local machine has NVIDIA-capable GPUs and you want to test your docker image locally using these GPUs then you will need to ensure the NVIDIA drivers have been
installed on your test machine.

Instructions on how to install the CUDA toolkit and NVIDIA drivers can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation). Be sure to match the versions of CUDA/NVIDIA installed with the version of PyTorch and CUDA used to build your docker image - see [Section 2: Building the docker image](#section-2-building-the-docker-image).

## 1.3 Clone this repository
Clone this repo to a machine which has docker installed:
```bash
git clone https://github.com/dmayo/objectnet_competition_demo.git
```

## 1.4 Running objectnet_eval.py
`objectnet_eval.py` is the main entry point for running this example.
Full help is available using `objectnet_eval.py --help`:
```bash
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
There follows a description of the code structure used in this repo.

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
- add your own model description class to this file.

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
argument ('input/images' in the example below) to the `objectnet_eval.py` program.

Then run `objectnet_eval.py` using the following arguments:
```bash
# Perform batch inference:
python objectnet_eval.py input/images output/predictions.csv resnext101_32x48d_wsl model/ig_resnext101_32x48-3e41cc8a.pth
```
Results will be written to the `predictions.csv` file in the `output/` directory. Check
the output conforms to the format expected by the ObjectNet Challenge **##### link to
format description**

## 1.7 Modifying the code to use your own PyTorch model

You can plugin your own existing existing PyTorch model into the test the process. As an example, the implementation of a pre-trained [InceptionV3 model](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py) is shown below:

#### 1.7.1 Requirements

This model uses the SciPy library. As this is not included in the default PyTorch Docker container it needs to be listed in the `requirements.txt` file so that it is 'pip installed' when the docker image is built. Include it as follows:
```bash
# This file specifies python dependencies which are to be installed into the Docker image.
# List one library per line (not as a comment)
# e.g.
#numpy
scipy
```
Uncomment the following line in the `Dockerfile`:
```bash
RUN pip install -r requirements.txt
```
#### 1.7.2 Model changes

The only code changes necessary when incorporating your PyTorch model should be in the `model/` directory.
1. Download your model checkpoint file into `model/`. For example:
```bash
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
```
2. Add your model description as a class to `./model/model_description.py`. The
class name will be used as the `model-class-name` argument to `objectnet_eval.py`.
For the inception_v3 model copy the `Inception3` class (along with any dependencies) from `inception.py` in the above link into `model_description.py`.
3. Amend the following parameters in `data_transformation_description.py` to match
those that your model was trained on:
```python
    input_size = [3, 299, 299]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
```
4. Test your model's inference using:
```bash
python objectnet_eval.py input/images output/predictions.csv Inception3 model/inception_v3_google-1a9a5a14.pth
```

## 1.8 Validating the predictions of your model
In order to ensure that the `predictions.csv` file is structured according to the ObjectNet Challenge specifications, it is important to validate it against the `validate_and_score.py` script.
Once your model has successfully executed run the following command to validate your output:
```bash
python validate_and_score.py -r -a input/answers/answers-test.json -f output/predictions.csv
```
Note the usage of the `-a` and `-f` flags as specified in `validate_and_score.py --help` below.
```
usage: validate_and_score.py [-h] --answers ANSWERS --filename FILENAME
                             [--range_check]
optional arguments:
  -h, --help            show this help message and exit
  --answers ANSWERS, -a ANSWERS
                        ground truth/answer file
  --filename FILENAME, -f FILENAME
                        users result file
  --range_check, -r     reject entries that have out-of-range label indices
{
  "prediction_file_errors": [
    "Failed to parse command line"
  ],
  "prediction_file_status": "INVALID"
}
```

Proceed to the next section if you receive an output of `"prediction_file_status": "VALIDATED"`.

If you received an error in running this command ensure that you have entered the correct file locations for the answer file as well as the result file. For clarification on result file structure refer to **####AS Insert link to spec**.


---


# Section 2 Building the docker image

## 2.1 Install NVIDIA drivers
Prior to uploading the docker image to the competition portal for evaluation you should test your docker image locally. If your local machine has NVIDIA-capable GPUs and you wish to test inference using GPUs then you will first need to install the NVIDIA drivers on your machine. See
section [1.2 Install NVIDIA drivers](#12-install-nvidia-drivers) above.

## 2.2 Add your model & supporting code
Ensure you have been able to successfully test your model on the local host using the `objectnet_eval.py` example code - see section [1.8 Validating the predictions of your model](#18-validating-the-predictions-of-your-model) for more details.

**#####SH is the below his true**

**Note:** Your model must have been saved using `torch.save(model, "<PATH TO SAVED MODEL FILE>")`.

## 2.3 Build the docker image
Docker images are built from a series of statements contained in a `Dockerfile`. A template Dockerfile is provided for models built using the PyTorch deep learning framework and saved using the `torch.save` api.

The PyTorch docker image template for the ObjectNet Challenge uses one of the [official PyTorch docker images](https://hub.docker.com/r/pytorch/pytorch/tags) as its base image. These PyTorch images come with built-in GPU support and with python 3 pre-loaded. By default, the docker image is built using [PyTorch version 1.4, cuda 10.1 and cudann7](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-runtime/images/sha256-ee783a4c0fccc7317c150450e84579544e171dd01a3f76cf2711262aced85bf7?context=explore).

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

A bash script, `build-docker-submission.sh`, has been created to build the Docker image
for you. The script has the following inputs:
```bash
This command runs builds your model into a Docker Image
Docker Image will be set to IMAGE:TAG

Default
NAME="resnext101_32x48d_wsl"
CHECKPOINT="model/ig_resnext101_32x48-3e41cc8a.pth"

options:
-h, --help			             	show brief help
-n, --model-class-name=NAME	    	specify a model class name to use
-c, --model-checkpoint=CHECKPOINT	specify the path to a model checkpoint to use
-i, --image=IMAGE		        	specify your Docker image
-t, --tag=TAG                       specify your Docker image tag
```
Create your image by running:
```bash
./build-docker-submission.sh -i IMAGE -t TAG -n NAME -c CHECKPOINT
```
where, for example
- IMAGE = my_model
- TAG = version1
- NAME = resnext101_32x48d_wsl
- CHECKPOINT = model/ig_resnext101_32x48-3e41cc8a.pth

**Note:** Please ensure you have no more than one checkpoint file in the `model\` directory when building the image. This will save space in the built Docker image.

Once the build is complete your newly built docker image can be listed using the command:
```bash
$ docker images
```

If the docker was built without version tagging it is given a default tag of `latest`.

## 2.4 Testing the docker image locally
Test the docker image locally before submitting it to the challenge. For example, a docker image called `my-model:version1` is run by:

```bash
docker run -ti --rm --gpus=all -v $PWD/input/images:/input/ -v $PWD/output:/output my-model:version1
```

The `-v $PWD/input/images:/input` mounts a directory of test images from the local path into `/input` within the docker container. Similarly, `-v $PWD/output:/output` mounts an output directory from the local path into `/output` of the container. Add the `--gpus=all` parameter to the `docker run` command in order to utilise your GPUs.

## 2.5 Debugging your docker image locally
If there are errors during the previous step then you will need to debug your docker container.
If you make changes to your code there is no need to rebuild the docker container. To quickly test your new code, simply mount the root path of this repo as a volume when you run the container. For example:
```bash
docker run -ti --rm --gpus=all -v $PWD:/workspace -v $PWD/input/images:/input/ -v $PWD/output:/output my-model:version1
```
When the docker container is run, the local `$PWD` will be mounted over `/workspace` directory within the docker image which effectively means any code/model changes made since the last `docker build` command will be contained within the running container.

## 2.6 Validating the predictions
In order to ensure that the `predictions.csv` file is structured according to the ObjectNet Challenge specifications, it is important to validate it against the `validate_and_score.py` script. Run the following command:
```bash
python validate_and_score.py -r -a input/answers/answers-test.json -f output/predictions.csv
```

Proceed to the next section if you receive an output of `"prediction_file_status": "VALIDATED"`. Otherwise, refer back to [1.8 Validating the predictions](#18-validating-the-predictions-of-your-model) to handle any errors.

---

# Upload your docker image to Synapse:
Once you have built and tested your docker image locally, refer to [Model Submission](https://www.synapse.org/#!Synapse:syn21445379/wiki/601749)**###AS Update to EvalAI** for instructions on uploading your image to the Synapse Docker registry and subsequent submission to the challenge.
