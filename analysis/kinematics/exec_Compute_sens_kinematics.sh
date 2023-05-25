#! /bin/sh

source activate qpe-firerisk

ADJ_FILE=${1}
INP_FILE=${2}
ADJ_OUTPUT_FILE=${3}
PYTHON_PROG=${4}

python3 ${PYTHON_PROG} << EOF
${ADJ_FILE}
${INP_FILE}
${ADJ_OUTPUT_FILE}
EOF

