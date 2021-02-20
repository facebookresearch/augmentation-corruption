These scripts run the main experiments of the paper. They are primarily 
for reference and should not be run as is: without setting up the proper 
parallelization using hydra and the submitit launcher, they will likely
time out or assign multiple jobs simultaneously to the same GPU. They 
also assume paths have been set in `conf/paths.yaml`. Random seed 
selection is slightly different here than in the original experiments: 
originally a custom hydra sweeper chose a different random number for 
every job in sweep; here, this is emulated by using the job's rank in 
the sweep.
