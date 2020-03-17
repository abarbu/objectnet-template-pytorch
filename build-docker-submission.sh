#!/usr/bin/env bash

NAME="resnext101_32x48d_wsl"
CHECKPOINT="model/ig_resnext101_32x48-3e41cc8a.pth"
ID=""
REPO=""
TAG="latest"
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "This command runs builds your model into a Docker Image"
      echo "Docker Image will be set to: docker.synapse.org/ID/REPO:TAG"
      echo ""
      echo "*Default*"
      echo "NAME=\"resnext101_32x48d_wsl\""
      echo "CHECKPOINT=\"model/ig_resnext101_32x48-3e41cc8a.pth\""
      echo ""
      echo "options:"
      echo "-h, --help				show brief help"
      echo "-n, --model-class-name=NAME		specify a model class name to use"
      echo "-c, --model-checkpoint=CHECKPOINT	specify the path to a model checkpoint to use"
      echo "-p, --project-id=ID			specify your Synapse Project ID"
      echo "-r, --repo=REPO			specify your repo name"
      echo "-t, --tag=TAG			specify your Docker image tag"
  exit 0
      ;;
    -c)
      shift
      if test $# -gt 0; then
        export CHECKPOINT=$1
      else
        echo "no model checkpoint specified"
        exit 1
      fi
      shift
      ;;
    --model-checkpoint*)
      export CHECKPOINT=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -n)
      shift
      if test $# -gt 0; then
        export NAME=$1
      else
        echo "no model class name specified"
        exit 1
      fi
      shift
      ;;
    --model-class-name*)
      export NAME=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -p)
      shift
      if test $# -gt 0; then
        export ID=$1
      else
        echo "no Synapse Project ID specified"
        exit 1
      fi
      shift
      ;;
    --project-id*)
      export ID=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -r)
      shift
      if test $# -gt 0; then
        export REPO=$1
      else
        echo "no repo name specified"
        exit 1
      fi
      shift
      ;;
    --repo)
      export REPO=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -t)
      shift
      if test $# -gt 0; then
        export TAG=$1
      else
        echo "no Docker image tag specified"
        exit 1
      fi
      shift
      ;;
    --tag*)
      export TAG=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    *)
      break
      ;;
  esac
done

# exit for errors
if [ "$ID" == "" ]; then
 echo "Error: no Synapse Project ID specified"
 echo ""
 exit 1
fi

if [ "$REPO" == "" ]; then
 echo "Error: no repo name specified"
 echo ""
 exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
 echo "Error: Checkpoint file does not exist at path \"$CHECKPOINT\""
 echo ""
 exit 1
fi

echo ""
echo "Using Arguments:"
echo "Model Class Name = $NAME"
echo "Model Checkpoint = $CHECKPOINT"
echo "Docker Image: docker.synapse.org/$ID/$REPO:$TAG"
echo ""

CHECKPOINT_FILE="${CHECKPOINT##*/}"

# check whether we're using the default model
if  [ "$CHECKPOINT_FILE" == "ig_resnext101_32x48-3e41cc8a.pth" ]; then
 # need to download the default checkpoint
 if [ ! -f "downloads/ig_resnext101_32x48-3e41cc8a.pth" ]; then
  echo "Downloading default model checkpoint"
  cd downloads
  wget https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth
  cd ..
 fi
 # add the checkpoint to the model directory if necessary
 if [ ! -f "model/ig_resnext101_32x48-3e41cc8a.pth" ]; then
  cp downloads/ig_resnext101_32x48-3e41cc8a.pth model
 fi; else
 # remove default checkpoint from model directory if not needed
 if [  -f "model/ig_resnext101_32x48-3e41cc8a.pth" ]; then
  rm model/ig_resnext101_32x48-3e41cc8a.pth
 fi
 cp $CHECKPOINT model
fi

docker build --build-arg MODEL_CLASS_NAME="$NAME" --build-arg MODEL_CHECKPOINT="$CHECKPOINT_FILE" -t docker.synapse.org/"$ID/$REPO:$TAG" -f Dockerfile .
