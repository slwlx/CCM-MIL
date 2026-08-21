from __future__ import print_function

import argparse
import os
os.environ["USE_TORCH"] = "1"       # 确保使用PyTorch后端

from timeit import default_timer as timer   # 用于精确记录训练耗时

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.survival_core_utils import train
from dataset.dataset_survival import Generic_MIL_Survival_Dataset

# pytorch imports
import torch
import pandas as pd
import numpy as np
import wandb

wandb.init(mode="offline")   # 仅本地记录，不弹出菜单

def main(args):
    print("args.mode =", args.mode)  # 输出：args.mode = ms-path

    # (1) 目录与实验初始化
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    # (2) W&B初始化
    wandb.init(project=args.task)
    wandb.config.update(args)

    # (3) K折交叉验证设置
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)   # 生成折数列表

    latest_test_cindex = []     # 存储测试集C-index
    latest_val_cindex = []      # 存储验证集C-index

    for i in folds:
        start = timer()         # 记录当前fold开始时间

        # (4) 固定随机种子确保可复现
        seed_torch(args.seed)   # 固定随机种子，保证可复现性
        # (5) 跳过已存在的结果
        # 若已存在结果文件，则跳过该 fold（避免重复训练），例子：results_pkl_path='./experiments/train/TCGA_BLCA_survival/mamba_mil/resnet50_s1/split_latest_val_0_results.pkl'
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue

        # (6) GPU内存清理（避免内存泄漏）
        # wlx：每次新的fold 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Fold {i+1}: GPU memory cleared before loading data.")

        # (7) 加载数据集（当前fold的分割文件）
        train_dataset, val_dataset, test_dataset = dataset.return_splits(args.backbone, args.patch_size, from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        # (8) 打印数据集大小
        if args.k_fold:
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        else: 
            print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

        # (9) 设置数据集元组
        if args.k_fold:
            datasets = (train_dataset, val_dataset)
        else:
            datasets = (train_dataset, val_dataset, test_dataset)

        # (10) 预加载数据到内存（加速训练）
        if args.preloading == 'yes':
            for d in datasets:
                d.pre_loading()     # 预加载特征

        # (11) 生存分析任务训练
        if args.task_type == 'survival':
            if args.k_fold:
                # 仅训练+验证（K折交叉验证）
                cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
            else:
                # 训练+验证+测试
                results, cindex_test, cindex_val = train(datasets, i, args)
                latest_val_cindex.append(cindex_val)
                latest_test_cindex.append(cindex_test)
            
        # results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        # all_test_auc.append(test_auc)
        # all_val_auc.append(val_auc)
        # all_test_acc.append(test_acc)
        # all_val_acc.append(val_acc)
        #write results to pkl
        # (12) 保存当前fold结果（非K折时保存测试结果）
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if not args.k_fold:
            save_pkl(filename, results)

    # (12.5) 如果所有 fold 都被跳过，直接返回
    if len(latest_val_cindex) == 0:
        print("All splits were skipped. Exiting.")
        return

    # (13) 汇总所有fold结果，汇总所有fold的C-index
    if args.k_fold:
        final_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    else: 
        final_df = pd.DataFrame({'folds': folds, 'test_cindex': latest_test_cindex, 
            'val_cindex': latest_val_cindex, })
    # (14) 保存汇总报告
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    # 保存为CSV
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    # (15) 计算平均C-index和标准差
    if not args.k_fold:
        mean_test = final_df['test_cindex'].mean()
        std_test = final_df['test_cindex'].std()
    mean_val = final_df['val_cindex'].mean()
    std_val = final_df['val_cindex'].std()
    # (16) 添加平均/标准差行
    if args.k_fold:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'val_cindex': [mean_val, std_val]
        })
    else:
        df_append = pd.DataFrame({
            'folds': ['mean', 'std'],
            'test_cindex': [mean_test, std_test],
            'val_cindex': [mean_val, std_val]
        })
    final_df = pd.concat([final_df, df_append])
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    # (17) 保存更新后的汇总报告
    final_df.to_csv(os.path.join(args.results_dir, save_name))

    # (18) 记录到W&B
    final_df['folds'] = final_df['folds'].astype(str)
    table = wandb.Table(dataframe=final_df)
    wandb.log({"summary": table})
    if args.k_fold:
        wandb.log({"mean_val_cindex": mean_val})
    else:
        wandb.log({"mean_test_cindex": mean_test, "mean_val_cindex": mean_val})

#位置：在 main() 函数之外，程序启动时执行
# ================== 参数解析 ==================
# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='Data directory to WSI features (extracted via CLAM)')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=1,)
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='enable dropout (p=0.25)')
parser.add_argument('--gc', type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['ccm_mil_v3_2'], default='ccm_mil_v3_2',
                    help='type of model')
# CCM specific arguments
parser.add_argument('--ccm_stage1_dir', type=int, default=4, choices=[1, 2, 4, 8], help='Number of directions in CCM Stage 1')
parser.add_argument('--ccm_stage2_mode', type=str, default='center_out', choices=['center_out', 'risk_gradient', 'structure_guided', 'rach', 'cbs', 'none'],
                    help='Semantic reordering mode for CCM Stage 2')
parser.add_argument('--ccm_soft_topk_ratio', type=float, default=0.3, help='Soft Top-K ratio for CCM Stage 2')
parser.add_argument('--ccm_stage2_layers', type=int, default=1, help='Number of Mamba2 layers in CCM Stage 2')
parser.add_argument('--ccm_drop_path_rate', type=float, default=0.0, help='Drop path rate for CCM Stage 1')
parser.add_argument('--ccm_diagonal_only', action='store_true', default=False,
                    help='Use only diagonal directions (45/135/225/315) when stage1_dir=4')
parser.add_argument('--ccm_selection_mode', type=str, default='soft', choices=['soft', 'hard'],
                    help="Stage-2 patch selection mode for CCM_MIL V3: 'soft' = reorder all patches, 'hard' = truncate to top-K")
parser.add_argument('--ccm_v3_grid_mode', type=str, default='square_norm', choices=['square_norm', 'aspect'],
                    help="Grid construction mode for CCM_MIL V3: 'square_norm' = legacy isotropic square grid; 'aspect' = aspect-preserving rectangular grid (backported from v5)")
parser.add_argument('--ablation_mode', type=str, default='none', choices=['none', 'no_stage1', 'no_stage2', 'no_lsmr', 'random_mask', 'random_order', 'grid_only', 'single_direction', 'no_selection', 'spread_reg', 'residual_s2', 'dir_attn', 'dual_head', 'spread_residual', 'spread_residual_dir', 'all'],
                    help='Ablation mode for CCM_MIL: none/full, no_stage1, no_stage2, no_lsmr (disable LSMR logits fusion), random_mask, random_order, spread_reg, residual_s2, dir_attn, dual_head, spread_residual, spread_residual_dir, all')

parser.add_argument('--mode', type = str, choices=['path', 'omic', 'pathomic', 'cluster', 'ms-path', 'ss-path'], default='path', help='which modalities to use')
parser.add_argument('--data_mode', type = str, choices=['ss-path', 'ms_patt'], default='ss-path', help='which modalities to use')
parser.add_argument('--apply_sig', action='store_true', default=False, help='Use genomic features as signature embeddings')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: None).')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str)
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--alpha_surv', type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--patch_size', type=str, default='')
parser.add_argument('--preloading', type=str, default='no')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--k_fold', type=bool, default=False, help='k fold for cross validation')
parser.add_argument('--use_h5', type=bool, default=False, help='k fold for cross validation')
# wlx在你的主函数或参数解析中添加，默认打开混合精度，最终保存的变量名是 args.use_amp
#parser.add_argument('--no_amp', action='store_false', dest='use_amp', default=True, help='Disable mixed precision training')
# wlx在你的主函数或参数解析中添加，默认关闭混合精度，最终保存的变量名是 args.use_amp
parser.add_argument('--use_amp', action='store_true', default=False, help='Enable mixed precision training (default: disabled)')

parser.add_argument('--patch_dim', type=str, default='')
parser.add_argument('--chunk_size',type=int, default=10000, help='chunk_size')
parser.add_argument('--save_epoch_every', type=int, default=0, help='Save intermediate checkpoint every N epochs (0=disabled)')

args = parser.parse_args()

# 设置设备（GPU优先）
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Deviece is:', device)
# 设置随机种子（确保可复现）
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

# args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
# 设置实验名称
print("Experiment Name:", args.exp_code)

# 通用设置
encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}


print('\nLoad Dataset')

# (19) 生存分析数据集初始化，全局数据集初始化（一次）
# 在 main() 函数之外，程序启动时执行
if 'survival' in args.task:
    args.n_classes = 4      # 生存分析任务4分类
    study = '_'.join(args.task.split('_')[:2])  # 提取癌症类型，如 TCGA_BLCA
    combined_study = study      # 如：BLCA

    # 数据目录：/dataset/pt_files/resnet50
    combined_study = combined_study.split('_')[1]
    # study_dir = '%s_20x_features' % combined_study
    study_dir = 'pt_files/%s' % args.backbone

    #dataset 是一个高级封装的数据集类实例
    # 本身不直接用于训练，而是作为一个“工厂”，后续通过其方法（如 return_splits）来生成具体的训练/验证/测试集
    dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/%s_processed.csv' % combined_study,
                                            mode = args.mode,
                                            use_h5 = args.use_h5,
                                            apply_sig = args.apply_sig,
                                            data_dir= os.path.join(args.data_root_dir, study_dir, args.patch_dim), #! cluster.pkl should be as same as data_dir
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,   # 生存时间分4个区间
                                            label_col = 'survival_months',  # 标签列
                                            ignore=[])
else:
	raise NotImplementedError

# 确认任务类型
if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError
    
# if not os.path.exists(args.results_dir):
#     os.mkdir(args.results_dir)

# (20) 设置结果目录（包含实验代码和种子）
args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

# (21) 设置分割目录
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)    # 确保分割目录存在

# (22) 保存实验设置
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)

# (23) 打印设置信息
print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start = timer()
    results = main(args)    # 启动训练
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))

