#! /bin/bash

condaBaseEnv=`conda info | grep -i 'base environment' | awk '{print $4}'`
source ${condaBaseEnv}/etc/profile.d/conda.sh

conda activate UWAOS_WRF_QOIP

# Define inputs:
#
# DATADIR: full-path to directory containing input files
# TMPLFIL: name of template file in local directory
# ITERBEG: beginning iteration (usually 0)
# ITEREND: ending iteration
# NCHUNKS: number of jobs to chunk into sub-jobs

DATADIR=/home/bhoover/UWAOS/WRF_QOIP/data_repository/case_archives/nov2019/R_mu/positive/uvTq
TMPLFIL=run_Compute_sens_kinematics.TMPL
ITERBEG=0
ITEREND=24
NCHUNKS=8

# use chunk_iter_list.py to generate strings of iterations for each sub-job chunk, pipe to file
python chunk_iter_list.py 0 24 8 > chunk_list.txt

# loop over chunks
n=0
while [ ${n} -lt ${NCHUNKS} ]
do
    # increment n
    ((n+=1))
    # copy template file for this sub-job to config.tmp
    cp ${TMPLFIL} config.tmp
    # extract nth line from chunk_list.txt
    ITERS=`awk NR==${n} chunk_list.txt`
    # replace >>DATADIR<< template space with value of DATADIR
    # NOTE: since ${DATADIR} contains '\' characters, which sed will interpret as its own
    #       delimiters, we will use the '^' command for sed delimiters instead, anticipating
    #       that no '^' characters exist in ${DATADIR}
    sed "s^>>DATADIR<<^${DATADIR}^g" config.tmp > tmp.file
    mv tmp.file config.tmp    
    # replace >>ITERLIST<< template space with value of ITERS
    sed "s/>>ITERLIST<</${ITERS}/g" config.tmp > tmp.file
    mv tmp.file config.tmp
    # rename config.tmp for sub-job and chmod to executable
    mv config.tmp ${TMPLFIL}.chunk${n}.sh
    chmod 700 ${TMPLFIL}.chunk${n}.sh
done

#for iter in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
#do
#
#adj_file=${data_dir}/gradient_wrfplus_d01_unpi${iter}
#inp_file=${data_dir}/wrfinput_d01_unpi${iter}
#adj_output_filename=gradient_kinematic_d01_unpi${iter}
#python_prog=Compute_sens_kinematics.py
#
#bash ./exec_Compute_sens_kinematics.sh ${adj_file} ${inp_file} ${adj_output_filename} ${python_prog}
#
#done
