#!/bin/bash
#SBATCH --partition=alldlc_gpu-rtx2080 #  partition (queue)
#SBATCH --mem 500000 # memory pool for all cores (8GB)
#SBATCH -t 01-00:00:00 # time (D-HH:MM)
#SBATCH -c 32 # number of cores
#SBATCH -a 1-10 # array size
#SBATCH -D /home/siemsj/projects/mothernet # Change working_dir
#SBATCH -o log_slurm/log_$USER_%Y-%m-%d.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log_slurm/err_$USER_%Y-%m-%d.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J baam  # sets the job name. If not specified, the file name will be used as job name
# #SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate conda environment
source ~/.bashrc
conda activate mothernet

for dataset_name in adult income churn support2 credit microsoft year
    do
        # Job to perform
        if [ $gpu_counter -eq $SLURM_ARRAY_TASK_ID ]; then
          PYTHONPATH=$PWD python mothernet/evaluation/benchmark_node_gam_datasets.py ${dataset_name}
          exit $?
        fi
    let gpu_counter+=1
done

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";