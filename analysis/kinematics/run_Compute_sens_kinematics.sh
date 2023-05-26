#! /bin/sh -f

data_dir=/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/positive/uvTq


#for iter in 00
for iter in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22
do

adj_file=${data_dir}/gradient_wrfplus_d01_unpi${iter}
inp_file=${data_dir}/wrfinput_d01_unpi${iter}
adj_output_filename=gradient_kinematic_d01_unpi${iter}
python_prog=Compute_sens_kinematics.py

bash ./exec_Compute_sens_kinematics.sh ${adj_file} ${inp_file} ${adj_output_filename} ${python_prog}

done
