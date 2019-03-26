import argparse

def str2bool(t):
    if t.lower() in ['true', 't', '1']:
        return True
    else:
        return False

def get_args():

    parser = argparse.ArgumentParser(description='EE 381V: Deep Learning for Case-Based Reasoning through Prototypes')

    # mode
    parser.add_argument('--mode', default='train_mnist', type=str, help='mode of the python script')

    # DataLoader
    parser.add_argument('--data_dir', default='./data/mnist', type=str, help='root directory of the dataset')
    parser.add_argument('--nworkers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--bsize', default=250, type=int, help='mini-batch size (default: 32)')
    parser.add_argument('--shuffle', default='True', type=str2bool, help='shuffle the data?')

    # Elastic Deformation Parameters
    parser.add_argument('--sigma', default=4, type=int, help='sigma value for the elastic deformation')
    parser.add_argument('--alpha', default=20, type=int, help='alpha value for the elastic deformation')

    # Model Parameters
    parser.add_argument('--n_prototypes', default=15, type=int, help='Number of prototypes in the network architecture')
    parser.add_argument('--dropout_p', default=0.2, type=float, help='Dropout probablity for the linear layers')

    # Optimization Parameters
    parser.add_argument('--optim', default='adam', type=str, help='Optimizer type')
    parser.add_argument('--lr', default=2e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epochs', default=1500, type=int, help='number of total epochs to run')
    parser.add_argument('--max_norm', default=1, type=float, help='Max grad norm')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    # Loss Term Parameters
    parser.add_argument('--lambda_class', default=20, type=int, help='weight proportion for the classification error')
    parser.add_argument('--lambda_ae', default=1, type=int, help='weight proportion for the auto-encoder error')
    parser.add_argument('--lambda_1', default=1, type=int, help='weight proportion for the interpretability regularization error')
    parser.add_argument('--lambda_2', default=1, type=int, help='weight proportion for the interpretability regularization error')

    # Save Parameter
    parser.add_argument('--save_path', default='./trained_models', type=str, help='Directory where models are saved')

    # Other
    parser.add_argument('--log_dir', default='./logs', type=str, help='Directory where tensorboardX logs are saved')
    parser.add_argument('--log_iter', default=100, type=int, help='print frequency (default: 10)')
    parser.add_argument('--resume', default=False, type=str2bool, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')

    args = parser.parse_args()

    return args
