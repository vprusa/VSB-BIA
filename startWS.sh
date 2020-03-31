#!/bin/bash

echo "Starting MA015 algorithms workspace"

~/IDE/pycharm-2018.2.4/bin/pycharm.sh & disown

PROJ_DIR=/home/vprusa/workspace/school/MUNI/MGR/3LS/MA015/proj/MA015-algorithms-samples
echo "In dir:"
echo "cd ${PROJ_DIR}"

printf "source .virtenv/bin/activate\n.virtenv/bin/ipython\n" 

cd ${PROJ_DIR}


#
