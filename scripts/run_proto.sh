source /scratch/cluster/pkar/pytorch-gpu-py3/bin/activate
code_root=__CODE_ROOT__

python -u $code_root/train.py \
	--mode __MODE__ \
	--data_dir __DATA_DIR__ \
	--nworkers __NWORKERS__ \
	--bsize __BSIZE__ \
	--shuffle __SHUFFLE__ \
	--sigma __SIGMA__ \
	--alpha __ALPHA__ \
	--n_prototypes __N_PROTOTYPES__ \
	--decoder_arch __DECODER_ARCH__ \
	--intermediate_channels __INTERMEDIATE_CHANNELS__ \
	--optim __OPTIM__ \
	--lr __LR__ \
	--wd __WD__ \
	--momentum __MOMENTUM__ \
	--epochs __EPOCHS__ \
	--max_norm __MAX_NORM__ \
	--start_epoch __START_EPOCH__ \
	--lambda_class __LAMBDA_CLASS__ \
	--lambda_ae __LAMBDA_AE__ \
	--lambda_1 __LAMBDA_1__ \
	--lambda_2 __LAMBDA_2__ \
	--save_path __SAVE_PATH__ \
	--log_dir __LOG_DIR__ \
	--log_iter __LOG_ITER__ \
	--resume __RESUME__ \
	--seed __SEED__
