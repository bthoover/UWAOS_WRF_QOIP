#! /bin/sh -f

data_dir=/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/positive/uvTq

for iter in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22
do

inp_file=${data_dir}/wrfinput_d01_unpi${iter}
ptd_file=${data_dir}/wrfinput_d01_ptdi${iter}
pert_output_filename=wrfinput_kinematic_d01_perti${iter}
python_prog=Compute_pert_kinematics.py

bash ./exec_Compute_pert_kinematics.sh ${inp_file} ${ptd_file} ${pert_output_filename} ${python_prog}

done
