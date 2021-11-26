#!/bin/bash

user=$(echo $USER)
postfix=""
PYTHON_VERSION=3.7
PYTHON_VERSION_SHORT=37
PYTHON_BIN=/usr/bin/python${PYTHON_VERSION}

function usage {
  printf "This script installs python.\n As arguments it takes:\m"
  printf "\t-u <username> ; to set ownership of dirs after install\n"
  printf "\t-p <postfix> ; to use in .vitrenv-<postfix> dir"
}

function parse_args {
  while getopts "u:p:" o; do
    case "${o}" in
    p)
      postfix="-${OPTARG}"
      ;;
    u)
      user=${OPTARG}
      ;;
    *)
      usage
      ;;
    esac
  done
  shift $((OPTIND - 1))
}

parse_args "$@"

# Install virtualenv, libcurl-devel, gcc, wget, unzipx
sudo yum install python python-virtualenv -y
# because of installation of python version 3.6
sudo yum install https://centos7.iuscommunity.org/ius-release.rpm -y
sudo yum install python${PYTHON_VERSION_SHORT}u python${PYTHON_VERSION_SHORT}u-pip python${PYTHON_VERSION_SHORT}u-devel -y

# Setup virtual environment
#virtualenv .virtenv${postfix}
rm -rf .virtenv${postfix}

virtualenv --python=${PYTHON_BIN} .virtenv${postfix}
source .virtenv${postfix}/bin/activate

# Install base requirements
pip${PYTHON_VERSION} install --upgrade setuptools
pip${PYTHON_VERSION} install ipython networkx Matplotlib disjoint-set seaborn

pip${PYTHON_VERSION} install -U pip

chown -R ${user}:${user} ./
echo -e "\nSetup Complete."

#
