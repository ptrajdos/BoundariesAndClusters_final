ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))
REQUIREMENTS_FILE=${ROOTDIR}/requirements.txt
REQUIREMENTS_FILE_PRE=${ROOTDIR}/requirements_pre.txt
DATAFILE=${ROOTDIR}/KeelData.tar.xz
DATAFILEID=1ef4amAkGcZKCkAu6Z2Vgq5jufLCRUGwA
DATADIR=${ROOTDIR}/data
VENV_SUBDIR=${ROOTDIR}/venv

MAVEN=mvn
PYTHON=python
PIP=pip

UNAME=$(shell uname -s)

ifeq ($(UNAME),Darwin)
	TAR=gtar
else
	TAR=tar
endif


.PHONY: all clean

create_env: create_venv maven_install download_data

run_bagging:
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${ROOTDIR}/src/main/python/experiments/bounds_and_clusters_bagging_experiment3.py

run_boosting:
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${ROOTDIR}./src/main/python/experiments/bounds_and_clusters_boosting_experiment3.py

maven_install:
	${MAVEN} clean install

create_venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install -r ${REQUIREMENTS_FILE_PRE};${PIP} install -r ${REQUIREMENTS_FILE}

download_data:
	curl -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"
	${TAR} -xvf ${DATAFILE} --wildcards --strip-components 4 --directory ${DATADIR} ./KeelData/datasetsKeel/datasetsArffBin/*.arff
	
