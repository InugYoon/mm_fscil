import argparse
import importlib
from utils_s import *


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default='bb')
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default='../ilfr/data/')


    # about pre-training
    #parser.add_argument('-epochs_base', type=int, default=1000) # original version
    parser.add_argument('-epochs_base', type=int, default=100)  # original version
    parser.add_argument('-epochs_new', type=int, default=1)
    parser.add_argument('-epochs_base_clf', type=int, default=1)
    parser.add_argument('-lr_base', type=float, default=0.1)
    #parser.add_argument('-lr_base', type=float, default=0.005)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-lr_new_enc', type=float, default=0.001)
    parser.add_argument('-lr_base_clf', type=float, default=5.0)
    parser.add_argument('-schedule', type=str, default='Milestone', # original version
                        choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-schedule_new', type=str, default='Milestone', # original version
                        choices=['Step', 'Milestone', 'Cosine'])
    #parser.add_argument('-milestones', nargs='+', type=int, default=[40, 70]) # mini
    parser.add_argument('-milestones', nargs='+', type=int, default=[50,100])  # mini
    parser.add_argument('-milestones_new', nargs='+', type=int, default=[50, 100])  # mini
    #parser.add_argument('-milestones', nargs='+', type=int, default=[100, 150])  # mini
    ###parser.add_argument('-milestones', nargs='+', type=int, default=[180, 210]) # For CUB200 pretrain=F
    parser.add_argument('-step', type=int, default=40)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    #parser.add_argument('-temperature', type=int, default=16)

    #parser.add_argument('-batch_size_base', type=int, default=4)
    parser.add_argument('-batch_size_base', type=int, default=128)
    #parser.add_argument('-batch_size_base', type=int, default=1)
    #parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    #parser.add_argument('-test_batch_size', type=int, default=1)


    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-load_dir', type=str, default=None) #load_dir & model_name should both exist or both not given.
    parser.add_argument('-model_name', type=str, default=None)
    #parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    #parser.add_argument('-obj_dir', type=str, default=OBJ_DIR)
    #parser.add_argument('-core_dir', type=str, default=CORE_DIR)

    # about training
    #parser.add_argument('-gpu', default='0,1')
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    #parser.add_argument('-num_workers', type=int, default=0)

    parser.add_argument('-log_path', type=str, default='results')
    #parser.add_argument('-seed', type=int, default=1, help='if 0: random. otherwise set seed.')
    parser.add_argument('-seed', type=int, default=1, help='if 0: random. otherwise set seed.')

    parser.add_argument('-base_class', type=int, default=60,
                        choices=[60,80,90,100,150,190,None])
    parser.add_argument('-way', type=int, default=5,
                        choices=[5,10,None])
    parser.add_argument('-shot', type=int, default=5)

    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new'
                             'use 0 or wanted value. Especially if shot is big'
                             'if way/shot is big, using 0 may occur CUDA memory error, '
                             'so set appropriate value for this e.g. 16')

    #parser.add_argument('-fw_mode', default='fc_cosface', choices=['ang_arcface', 'ang_cosface', 'ang_ce',
     #parser.add_argument('-base_doubleaug', default=True)#parser.add_argument("-base_freeze_backbone", default=False)
    parser.add_argument("-base_freeze_backbone", action='store_true')
    parser.add_argument("-inc_freeze_backbone", action='store_true')
    parser.add_argument('-base_doubleaug', action='store_true') # for supcon
    parser.add_argument('-inc_doubleaug', action='store_true')
    parser.add_argument('-angle_exp', action='store_true')
    parser.add_argument('-plot_tsne', action='store_true')
    parser.add_argument('-rpclf_normmean', action='store_true')
    parser.add_argument('-use_cifar_resnet18', action='store_true')
    parser.add_argument('-use_cifar_resnet18_opt1', action='store_true')
    parser.add_argument('-use_cifar_resnet18_opt2', action='store_true')
    parser.add_argument('-use_cifar_resnet18_opt3', action='store_true')
    parser.add_argument('-use_cifar_resnet18_mini', action='store_true')
    parser.add_argument('-use_supcon_resnet18', action='store_true')
    parser.add_argument('-no_rbfc', action='store_true')
    parser.add_argument('-rbfc_opt2', action='store_true')

    #parser.add_argument('-plot_tsne', default = True)
    # way is determined by dataset.
    parser.add_argument('-base_dataloader_mode', default='plain', # for cs-kd
                        choices=['plain', 'episodic', 'pair'])
    parser.add_argument('-inc_dataloader_mode', default='plain', # for cs-kd
                        choices=['plain', 'episodic', 'pair'])

    parser.add_argument('-use_celoss', action='store_true')
    parser.add_argument('-fw_mode', default='fc_cos', choices=['ang_arcface', 'ang_cosface', 'ang_ce',
                                                               'fc_dot', 'fc_cos', 'fc_cosface'])  # forward mode
    parser.add_argument('-s', type=float, default=16.0)  # temperature
    parser.add_argument('-m', type=float, default=0.0)

    parser.add_argument('-use_supconloss', action='store_true')
    parser.add_argument('-supcontemp', default=0.07, type=float, help='used in supconloss')
    parser.add_argument('-supcon_angle', action='store_true')
    parser.add_argument('-aug_type1', type=int, nargs='+',
                        default=[1, 3, 4])  # default [1,2,3,4] for cifar, [1,2] for cub.
    # for mini, order is [1,3,2] where colorjitter(2) is fixed instead of randomapply
    # 0 means no aug, 1~4: aug types
    parser.add_argument('-aug_type2', type=int, nargs='+', default=[1, 3, 4])  # originally aug_types2 was equal to one

    parser.add_argument('- ', action='store_true') #original ver, with pair setting
    parser.add_argument('-use_cskdloss_1', action='store_true') # for data augment version
    parser.add_argument('-use_cskdloss_2', action='store_true') # for data augment version
    parser.add_argument('--lamda', default=5.0, type=float, help='cls loss weight ratio')

    #parser.add_argument('-cskd_opt1', action='store_true')
    #parser.add_argument('-cskd_opt2', action='store_true')

    parser.add_argument('-save_freq', type=int, default=100)
    parser.add_argument('-base_clf_ft', action='store_true')

    parser.add_argument('-gauss', action='store_true')
    parser.add_argument('-tukey_beta', type=float, default=0.5)
    parser.add_argument('-num_sampled', type=int, default=10)

    parser.add_argument('-use_encmlp', action='store_true')
    parser.add_argument('-encmlp_dim', type=int, default=256)
    parser.add_argument('-encmlp_layers', type=int, default=2)

    parser.add_argument('-use_head', action='store_true')
    parser.add_argument('-head_type', type=str, default='mlp', choices=['mlp', 'lin', 'none'])
    parser.add_argument('-head_dim', type=int, default=256)

    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer()
    trainer.main(args)