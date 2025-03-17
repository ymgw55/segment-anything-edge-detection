#!/bin/bash

# Check if an argument is provided, if not display usage and exit
if [ -z "$1" ]; then
  echo "Usage: $0 <gpu-device-id>"
  exit 1
fi

project_name="samed"
device_id=$1
container_name="${project_name}_${device_id}"
docker_image="${USER}/${project_name}"

# set DOCKER_HOME to your desired home directory in the container
# DOCKER_HOME=path/to/your/home

docker run --rm -it --name ${container_name} \
-u $(id -u):$(id -g) \
--gpus device=${device_id} \
-v ${DOCKER_HOME}:/home/${USER} \
-v $PWD:/working \
${docker_image} bash