#!/bin/bash
source activate falconcv
/falconcv/docker/start_jupyter.sh > /dev/null
exec "$@"