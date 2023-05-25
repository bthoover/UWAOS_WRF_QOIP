#! /bin/sh -f

data_dir=/Volumes/R10N_STORE/SSEC_SAVE/WRF_QOIP/case_archives/march2020/R_mu/negative/uvTq

for iter in 00 01 02 03 04 05 06 07 08 09 10 11 12 13
do

inp_file=${data_dir}/wrfinput_d01_unpi${iter}
ptd_file=${data_dir}/wrfinput_d01_ptdi${iter}
pert_output_filename=wrfinput_kinematic_d01_perti${iter}
python_prog=Compute_pert_kinematics.py

bash ./exec_Compute_pert_kinematics.sh ${inp_file} ${ptd_file} ${pert_output_filename} ${python_prog}

done
