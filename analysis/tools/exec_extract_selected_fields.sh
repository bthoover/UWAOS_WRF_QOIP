#! /bin/sh

source activate UWAOS_WRF_QOIP

baseDataDir=${1}
Hires=${2}
YYYY=${3}
MM=${4}
DD=${5}
HH=${6}
fcstHr=${7}
PYTHON_PROG=${8}

python3 ${PYTHON_PROG} ${baseDataDir} ${Hires} ${YYYY} ${MM} ${DD} ${HH} ${fcstHr}


