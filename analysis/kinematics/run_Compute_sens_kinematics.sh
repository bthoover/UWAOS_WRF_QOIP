#! /bin/sh -f

data_dir=/Volumes/R10N_STORE/SSEC_SAVE/WRF_QOIP/case_archives/march2020/R_mu/negative/uvTq


for iter in 00 01 02 03 04 05 06 07 08 09 10 11 12 13
do

adj_file=${data_dir}/gradient_wrfplus_d01_unpi${iter}
inp_file=${data_dir}/wrfinput_d01_unpi${iter}
adj_output_filename=gradient_kinematic_d01_unpi${iter}
python_prog=Compute_sens_kinematics.py

bash ./exec_Compute_sens_kinematics.sh ${adj_file} ${inp_file} ${adj_output_filename} ${python_prog}

done
