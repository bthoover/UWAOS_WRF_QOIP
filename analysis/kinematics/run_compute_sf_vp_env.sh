#! /bin/sh -f

baseDataDir=/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Ida2021/R_mu/unperturbed/
Lores=27km
Hires=9km
YYYY=2021    #Ida2021: YYYY=2021    Michael2018: YYYY=2018
MM=08        #Ida2021: MM=08        Michael2018: MM=10
DD=28        #Ida2021: DD=28        Michael2018: DD=9
HH=18        #Ida2021: HH=18        Michael2018: HH=12
start_i=20   #Ida2021: start_i=20   Michael2018: start_i=60
start_j=15   #Ida2021: start_j=15   Michael2018: start_j=15
resRatio=3
PYTHON_PROG=compute_sf_vp_env.py


for F in 1 2 3 4
do
    fcstHr=${F}
    bash ./exec_compute_sf_vp_env.sh ${baseDataDir} ${Lores} ${Hires} ${YYYY} ${MM} ${DD} ${HH} ${fcstHr} ${start_i} ${start_j} ${resRatio} ${PYTHON_PROG}
done
