import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--cr', default=0.8, type=float, help='contrastive loss rate')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=30, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--gcn_bool', action='store_true', default=True, help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', default=False, help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', default=True, help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true', default=False, help='whether random initialize adaptive adj')
    parser.add_argument('--temporalRange', default=30, type=int, help='number of hops for temporal features')
    parser.add_argument('--data', default='CHI', type=str, help='name of dataset') #NYC,CHI
    parser.add_argument('--kernelSize', default=3, type=int, help='size of kernel')
    parser.add_argument('--border', default=0.5, type=float, help='border line for pos and neg predictions')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
    parser.add_argument('--dropRate', default=0.0, type=float, help='drop rate for dropout')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--save', type=str, default='./Save/', help='save path')
    parser.add_argument('--checkpoint', type=str, default='./Save/CHI/', help='test path')

    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')

    return parser.parse_args()
args = parse_args()