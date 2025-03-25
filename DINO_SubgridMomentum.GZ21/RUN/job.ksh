#!/bin/ksh
######################
## JEANZAY IDRIS ##
######################
#SBATCH --job-name=DINO.GZ21
#SBATCH --output=DINO.GZ21.out
#SBATCH --error=DINO.GZ21.err
#SBATCH --ntasks=2
#SBATCH --hint=nomultithread # One MPI process per physical core (no hyperthreading)
#SBATCH --time=11:30:00
#SBATCH --account=cli@cpu
#SBATCH --qos=qos_cpu-dev # Queue test

# Process distribution
NPROC_NEMO=1
NPROC_PYTHON=1

## -------------------------------------------------------
##   End of user-defined section - modify with knowledge
## -------------------------------------------------------
# Load Environnment
source ~/.bash_profile

# Move to execution directory
cd ${SLURM_SUBMIT_DIR}
set -x
pwd

# job information 
cat << EOF
------------------------------------------------------------------
Job submit on $SLURM_SUBMIT_HOST by $SLURM_JOB_USER
JobID=$SLURM_JOBID Running_Node=$SLURM_NODELIST 
Node=$SLURM_JOB_NUM_NODES Task=$SLURM_NTASKS
------------------------------------------------------------------
EOF

# Begin of section with executable commands
set -e
ls -l

# run eophis in preproduction mode to generate namcouple
touch namcouple
rm namcouple*
python3 ./main.py --exec preprod

# save eophis preproduction logs
mv eophis.out eophis_preprod.out
mv eophis.err eophis_preprod.err

# check if preproduction did well generate namcouple
namcouple=namcouple
if [ ! -e ${namcouple} ]; then
        echo "namcouple can not be found, preproduction failed"
        exit 1
else
        echo "preproduction successful"
fi

# write multi-prog file
touch run_file
rm run_file
echo 0-$((NPROC_NEMO - 1)) ./nemo >> run_file
echo ${NPROC_NEMO}-$((NPROC_NEMO + NPROC_PYTHON - 1)) python3 ./main.py >> run_file

# run coupled NEMO-Python
time srun --multi-prog ./run_file
