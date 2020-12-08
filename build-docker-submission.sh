#!/usr/bin/env bash

if [ "$0" != "./build-docker-submission.sh" ] && [ "$0" != "build-docker-submission.sh" ]; then
  echo -e "\nError: $0" >&2
  echo "This script must be run from the base folder of this repo" >&2
  exit 1
fi
 
NAME=""
CHECKPOINT=""
IMAGE=""
TAG="latest"
CACHE=true
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "This command runs builds your model into a Docker Image"
      echo "Docker Image will be set to IMAGE:TAG"
      echo ""
      echo -e "\e[3mDefault\e[0m"
      echo "TAG=\"latest\""
      echo ""
      echo "options:"
      echo "-h, --help				show brief help"
      echo "-n, --model-class-name=NAME		specify a model class name to use"
      echo "-c, --model-checkpoint=CHECKPOINT	specify the path to a model checkpoint to use"
      echo "-i, --image=IMAGE			specify your Docker image"
      echo "-t, --tag=TAG			specify your Docker image tag"
      echo "-nc, --no-cache			bypass cache for docker build"
      exit 0
      ;;
    -c)
      shift
      if test $# -gt 0; then
        export CHECKPOINT=$1
      else
        echo "Error: no model checkpoint specified"
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
        echo "Error: no model class name specified"
        exit 1
      fi
      shift
      ;;
    --model-class-name*)
      export NAME=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -i)
      shift
      if test $# -gt 0; then
        export IMAGE=$1
      else
        echo "Error: no Docker image specified"
        exit 1
      fi
      shift
      ;;
    --image*)
      export IMAGE=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    -t)
      shift
      if test $# -gt 0; then
        export TAG=$1
      else
        echo "Error: no Docker image tag specified"
        exit 1
      fi
      shift
      ;;
    --tag*)
      export TAG=`echo $1 | sed -e 's/^[^=]*=//g'`
      shift
      ;;
    --no-cache | -nc)
      export CACHE=false
      shift
      ;;
    *)
      echo "Error: flag $1 does not exist"
      exit 1
      ;;
  esac
done

# exit for errors
if [ "$IMAGE" == "" ]; then
 echo "Error: no Docker image specified"
 echo ""
 exit 1
fi
if [ "$NAME" == "" ]; then
 echo "Error: no model class name specified"
 echo ""
 exit 1
fi
if [ "$CHECKPOINT" == "" ]; then
 echo "Error: no model checkpoint file path specified"
 echo ""
 exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
 echo "Error: checkpoint file does not exist at path \"$CHECKPOINT\""
 echo ""
 exit 1
fi

echo ""
echo "Using Arguments:"
echo "Model Class Name = $NAME"
echo "Model Checkpoint = $CHECKPOINT"
echo "Docker Image: $IMAGE:$TAG"
echo "Cache: $CACHE"
echo ""

CHECKPOINT_FILE="${CHECKPOINT##*/}"
if [ "$CACHE" == true ]; then
 docker build --build-arg MODEL_CLASS_NAME="$NAME" --build-arg MODEL_CHECKPOINT="$CHECKPOINT_FILE" -t "$IMAGE:$TAG" -f Dockerfile .
else
 docker build --no-cache --build-arg MODEL_CLASS_NAME="$NAME" --build-arg MODEL_CHECKPOINT="$CHECKPOINT_FILE" -t "$IMAGE:$TAG" -f Dockerfile .
fi

