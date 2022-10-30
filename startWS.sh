#!/bin/bash

echo "Starting workspace"

~/workspace/IDE/pycharm-community-2022.2.3/bin/pycharm.sh & disown

#PROJ_DIR=/home/vprusa/workspace/school/MUNI/MGR/3LS/MA015/proj/MA015-algorithms-samples
PROJ_DIR=/home/vprusa/workspace/school/VSB/1ZS/BIA-Biologicky_inspirovane_algoritmy/ukoly/1/1

echo "In dir:"
echo "cd ${PROJ_DIR}"

printf "source .virtenv-local/bin/activate\n.virtenv-local/bin/ipython\n" 

cd ${PROJ_DIR}


#
