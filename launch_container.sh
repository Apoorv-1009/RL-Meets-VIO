#!/bin/sh
xhost local:root

XAUTH=/tmp/.docker.xauth

docker run --privileged --rm -it \
    --volume /home/vinaylanka/Desktop/semester/rl_vo/rl_vo_docker_home/vo_rl/:/home/vinaylanka/vo_rl/:rw \
    --volume /home/vinaylanka/Desktop/semester/rl_vo/rl_vo_docker_home/logs/log_voRL/training/:/home/vinaylanka/logs/log_voRL/training/:rw \
    --volume /home/vinaylanka/Desktop/semester/rl_vo/rl_vo_docker_home/datasets/EuRoC/:/home/vinaylanka/datasets/EuroC/:ro \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --ipc=host \
    --privileged \
    --user $(id -u):$(id -g) \
    --gpus=all \
    vo_rl
    bash
