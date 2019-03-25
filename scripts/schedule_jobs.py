import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/pytorch-prototypeDL/runs/cifar'
SCRIPT_ROOT='/scratch/cluster/pkar/pytorch-prototypeDL/scripts'

mapping_dict = {
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL',
    # Script parameters
    '__JOBNAME__': ['cifar_lr_2e-3', 'cifar_lr_6e-4', 'cifar_lr_6e-3'],
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/pytorch-prototypeDL',
    '__MODE__': 'train_cifar',
    '__DATA_DIR__': '/scratch/cluster/pkar/pytorch-prototypeDL/data/cifar',
    '__NWORKERS__': '4',
    '__BSIZE__': '250',
    '__SHUFFLE__': 'True',
    '__SIGMA__': '4',
    '__ALPHA__': '20',
    '__N_PROTOTYPES__': '15',
    '__DROPOUT_P__': '0.2',
    '__OPTIM__': 'adam',
    '__LR__': ['2e-3', '6e-4', '6e-3'],
    '__WD__': '4e-5',
    '__MOMENTUM__': '0.9',
    '__EPOCHS__': '1500',
    '__MAX_NORM__': '1',
    '__START_EPOCH__': '0',
    '__LAMBDA_CLASS__': '20',
    '__LAMBDA_AE__': '1',
    '__LAMBDA_1__': '1',
    '__LAMBDA_2__': '1',
    '__LOG_ITER__': '100',
    '__RESUME__': 'False',
    '__SEED__': '123',
    }

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.items():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    job_name = mapping_dict['__JOBNAME__'][idx]
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, job_name)
    if os.path.isdir(mapping_dict['__LOGNAME__']):
        print ('Skipping job ', mapping_dict['__LOGNAME__'], ' directory exists')
        continue

    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.items():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
