#! /bin/sh

source activate UWAOS_WRF_QOIP

baseDataDir=${1}
Lores=${2}
Hires=${3}
YYYY=${4}
MM=${5}
DD=${6}
HH=${7}
fcstHr=${8}
start_i=${9}
start_j=${10}
resRatio=${11}
PYTHON_PROG=${12}

python3 ${PYTHON_PROG} ${baseDataDir} ${Lores} ${Hires} ${YYYY} ${MM} ${DD} ${HH} ${fcstHr} ${start_i} ${start_j} ${resRatio}


