#! /bin/bash

OUTCHAR=
INCHAR=:

echo "SWAPPING ${OUTCHAR} WITH ${INCHAR}..."
ls -1 *${OUTCHAR}* > swap_list.txt
NFILES=`wc -l swap_list.txt | awk '{print $1}'`
let N=0
while [ ${N} -lt ${NFILES} ]; do
	N=$(( ${N} + 1 ))
	OUTFILE=`awk "NR==${N}" swap_list.txt`
	INFILE=`echo "${OUTFILE}" | sed -r "s/[${OUTCHAR}]+/${INCHAR}/g"`
	echo "${OUTFILE} to ${INFILE}"
	mv ${OUTFILE} ${INFILE}
done
rm swap_list.txt
