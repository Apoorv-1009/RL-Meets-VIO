#!/bin/sh
xhost local:root


XAUTH=/tmp/.docker.xauth

docker run --privileged --rm -it \
    --volume /home/apoorv/rl_vio/rl_vo/:/home/apoorv/rl_vio/rl_vo/vo_rl/:rw \
    --volume /home/apoorv/rl_vio/rl_vo/log_voRL/:/home/apoorv/logs/log_voRL/:rw \
    --volume /home/apoorv/rl_vio/EuRoC/:/home/apoorv/datasets/EuRoC/:ro \
    --volume /home/apoorv/rl_vio/TartanAir/:/home/apoorv/datasets/TartanAir/:ro \
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