#!/bin/sh
#PBS -N qsub_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=2:ppn=4

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    scp master_ubss1:/home/${USER}/svd/main ${node}:/home/${USER}/main 1>&2
    scp -r master_ubss1:/home/${USER}/svd/files ${node}:/home/${USER}/ 1>&2
done

cd /home/${USER}

rm -f /home/${USER}/files/mpi_profile_np*.csv

# 最多运行 30 分钟，超过后自动终止，防止 MPI 死锁长期占用节点。
timeout 30m /usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main 20260410
STATUS=$?

scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/svd/ 2>&1

exit $STATUS