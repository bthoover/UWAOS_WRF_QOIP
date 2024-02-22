#! /bin/sh -f

baseDataDir=/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/
Lores=27km
Hires=9km
YYYY=2021
MM=8
DD=28
HH=18
fcstHr=0
start_i=20
start_j=15
resRatio=3
PYTHON_PROG=compute_sf_vp.py


for F in 6 12 18
do
    fcstHr=${F}
    bash ./exec_compute_sf_vp.sh ${baseDataDir} ${Lores} ${Hires} ${YYYY} ${MM} ${DD} ${HH} ${fcstHr} ${start_i} ${start_j} ${resRatio} ${PYTHON_PROG}
done
