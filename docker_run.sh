#!/usr/bin/env bash

## Fail if any command fails (use "|| true" if a command is ok to fail)
set -e
## Treat unset variables as error
set -u

dummy=`which nvidia-docker`;
if test $? -eq 0; then
  DOCKER_CMD='nvidia-docker run';
else
  DOCKER_CMD='docker run --runtime=nvidia';
fi

## Setup X authority such that the container knows how to do graphical stuff
XSOCK="/tmp/.X11-unix";
XAUTH=`tempfile -s .docker.xauth`;
xauth nlist "${DISPLAY}"          \
  | sed -e 's/^..../ffff/'        \
  | xauth -f "${XAUTH}" nmerge -;

PWD="/misc/lmbraid18/zimmermc/projects/FreiPose/tmp/"  # TODO remove this

${DOCKER_CMD}                     \
  --rm                            \
  --volume "${XSOCK}:${XSOCK}:rw" \
  --volume "${XAUTH}:${XAUTH}:rw" \
  --env "XAUTHORITY=${XAUTH}"     \
  --env DISPLAY                   \
   --volume "${PWD}/data/:/host/:rw"        \
  --hostname "${HOSTNAME}"        \
  --env QT_X11_NO_MITSHM=1 \
-it docker-freipose /bin/bash;


