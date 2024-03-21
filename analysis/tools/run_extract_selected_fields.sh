#! /bin/sh -f

baseDataDir=/home/bhoover/UWAOS/WRF_QOIP/data_repository/final_runs/Michael2018/R_mu/negative/uvTq/upper_troposphere/ptdi08/
Hires=9km
YYYY=2018
MM=10
DD=9
HH=12
PYTHON_PROG=extract_selected_fields.py


for F in 0
do
    fcstHr=${F}
    bash ./exec_extract_selected_fields.sh ${baseDataDir} ${Hires} ${YYYY} ${MM} ${DD} ${HH} ${fcstHr} ${PYTHON_PROG}
done
