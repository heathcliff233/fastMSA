CDIR=$1
ODIR=$2
qjackhmmer=/share/hongliang/qjackhmmer
for file in /share/hongliang/speed-test/casp14-query/*
  do
    fname=${file##*/}
    $qjackhmmer -B ${ODIR}${fname} -E 0.001 --cpu 4 -N 3 ${file} ${CDIR}${fname} > /dev/null
    #$qjackhmmer -B ${ODIR}${fname} -E 0.001 --cpu 4 -N 3 ${file} /share/hongliang/res-database.fasta > /dev/null
  done
