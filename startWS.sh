#!/bin/bash

echo "Starting MA015 algorithms workspace"

~/IDE/pycharm-2018.2.4/bin/pycharm.sh & disown

PROJ_DIR=/home/vprusa/workspace/school/MUNI/MGR/3LS/MA015/proj/MA015-algorithms-samples
echo "In dir:"
echo ${PROJ_DIR}

printf "Exec: \nsource .virtenv/bin/activate\n.virtenv/bin/iptyhon\n" 

cd ${PROJ_DIR}


#
