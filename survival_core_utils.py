from argparse import Namespace  # Namespace: 用于处理命令行参数
from collections import OrderedDict  # OrderedDict: 有序字典
import os  # os: 操作系统相关功能（如文件路径操作）

os.environ["USE_TORCH"] = "1"
import pickle  # pickle: 对象序列化

from lifelines.utils import concordance_index  # concordance_index 和 concordance_index_censored: 计算生存分析中的 C-index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch  # torch: PyTorch 核心库

from dataset.dataset_generic import save_splits  # save_splits: 保存数据分割结果（自定义函数）
from utils.survival_utils import *  # urvival_utils: 自定义生存分析工具函数
import wandb  # wandb: 实验跟踪工具


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    '''在验证损失不再改善时提前停止训练，防止过拟合。'''

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup  # 预热期 (warmup): 初始若干 epoch 不进行早停判断
        self.patience = patience  # 容忍期 (patience): 连续多少个 epoch 无改善则停止
        self.stop_epoch = stop_epoch  # 早停条件: 计数器达到 patience 且当前 epoch 超过 stop_epoch
        self.verbose = verbose  # 是否打印信息
        self.counter = 0  # 无改善的 epoch 计数器
        self.best_score = None  # 最佳验证分数
        self.early_stop = False  # 是否触发早停
        self.val_loss_min = np.Inf  # 最小验证损失

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass  # 预热期不处理
        elif self.best_score is None:
            self.best_score = score  # 初始化最佳分数
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1  # 无改善，计数器+1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:  # 早停条件: 计数器达到 patience 且当前 epoch 超过 stop_epoch
                self.early_stop = True  # 触发早停
        else:
            self.best_score = score  # 更新最佳分数
            self.save_checkpoint(val_loss, model, ckpt_name)  # 模型保存: 每次验证损失改善时保存模型
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        '''保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class EarlyStopping_cindex:
    """Early stops the training if validation C-index doesn't improve after a given patience."""
    '''
    专用于 C-index 指标（生存分析）的早停。
    逻辑: C-index 越大越好，持平不触发 counter（使用 < 而非 <=）
    '''

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_cindex = -np.Inf

    def __call__(self, epoch, val_cindex, model, ckpt_name='checkpoint.pt'):

        score = val_cindex

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_cindex, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, model, ckpt_name):
        '''Saves model when validation C-index increases.'''
        if self.verbose:
            print(f'Validation C-index increased ({self.best_cindex:.6f} --> {val_cindex:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.best_cindex = val_cindex


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = 'checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:  # 若当前 C-index 高于历史最佳，则保存模型。
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train(datasets: tuple, cur: int, args: Namespace):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))

    print(f"Using device: {device}")

    # 日志与结果目录设置
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    # 数据分割处理
    if args.k_fold:
        # 如果是K折交叉验证（args.k_fold），datasets仅包含训练集和验证集
        print('K-fold cross validation')
        print('----------------K-fold cross validation---------------------------------------', args.k_fold)
        train_split, val_split = datasets
        # print('train_split.shape, val_split.shape',train_split.shape, val_split.shape)
        # print(val_split)
    else:
        # 否则，分割为训练集、验证集和测试集，并保存分割结果到CSV文件。
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print('\nInit loss function...', end=' ')

    # 损失函数初始化：根据任务类型和配置初始化损失函数
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':  # 交叉熵生存损失（ce_surv）
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':  # 负对数似然生存损失（nll_surv）
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':  # Cox比例风险损失（cox_surv）
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # 正则化函数初始化：根据正则化类型选择正则化函数。
    if args.reg_type == 'omic':  # omic: 对所有参数应用L1正则化。
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':  # pathomic: 对特定模块应用L1正则化。
        reg_fn = l1_reg_modules
    else:  # 默认不启用正则化。
        reg_fn = None

    print('Done!')

    # 模型初始化
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    # 根据args.model_type动态导入并初始化模型
    if args.model_type == 'ccm_mil_v3_2':
        from models.ccm_mil_v3_2 import CCM_MIL
        model = CCM_MIL(in_dim=args.in_dim, n_classes=args.n_classes, dropout=args.drop_out, act='gelu',
                        survival=True, stage1_dir=getattr(args, 'ccm_stage1_dir', 4),
                        stage2_mode=getattr(args, 'ccm_stage2_mode', 'center_out'),
                        soft_topk_ratio=getattr(args, 'ccm_soft_topk_ratio', 0.3),
                        stage2_layers=getattr(args, 'ccm_stage2_layers', 1),
                        drop_path_rate=getattr(args, 'ccm_drop_path_rate', 0.0),
                        ablation_mode=getattr(args, 'ablation_mode', 'none'),
                        selection_mode=getattr(args, 'ccm_selection_mode', 'soft'),
                        diagonal_only=getattr(args, 'ccm_diagonal_only', False),
                        grid_mode=getattr(args, 'ccm_v3_grid_mode', 'square_norm'))
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))

    print('Init Model Done!')
    print(f'print {args.model_type} Model:')
    print_network(model)

    # 优化器初始化
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    # wlx添加：初始化 GradScaler（仅当使用 AMP 时）
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("wlx添加：初始化 GradScaler（仅当使用 AMP 时）:true")
    else:
        scaler = None
        print("wlx添加：初始化 GradScaler（仅当使用 AMP 时）:false")

    # 数据加载器初始化
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                    weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    if not args.k_fold:
        test_loader = get_split_loader(test_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    # 早停机制初始化
    # 如果启用早停（args.early_stopping），根据是否为K折选择不同的早停类（EarlyStopping_cindex或EarlyStopping）
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.k_fold:
            early_stopping = EarlyStopping_cindex(warmup=0, patience=20, stop_epoch=40, verbose=True)
        else:
            early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=40, verbose=True)
    else:
        early_stopping = None

    # 验证指标监控初始化
    # 初始化C-index监控器（用于生存分析任务的评估）
    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    # V4: Cox-PH auxiliary loss buffer
    cox_buffer = []
    cox_aux_weight = getattr(args, 'cox_aux_weight', 0.0)
    cox_buffer_size = getattr(args, 'cox_buffer_size', 8)

    # 训练循环 epoch
    for epoch in range(args.max_epochs):
        # 对每个epoch
        if args.task_type == 'survival':
            # 调用train_loop_survival执行生存分析任务的训练。
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn,
                                args.lambda_reg, args.gc, scaler=scaler,
                                chunk_size=args.chunk_size if hasattr(args, 'chunk_size') else 10000,
                                cox_buffer=cox_buffer, cox_aux_weight=cox_aux_weight, cox_buffer_size=cox_buffer_size)
            # 调用validate_survival执行验证并检查早停条件
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex,
                                     writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.k_fold,
                                     chunk_size=args.chunk_size if hasattr(args, 'chunk_size') else 10000)

        # Save intermediate checkpoint every N epochs (for attention visualization across epochs)
        save_every = getattr(args, 'save_epoch_every', 0)
        if save_every > 0 and epoch % save_every == 0 and args.results_dir is not None:
            epoch_ckpt_dir = os.path.join(args.results_dir, 'epoch_checkpoints')
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            epoch_ckpt_path = os.path.join(epoch_ckpt_dir, "s_{}_epoch_{}.pt".format(cur, epoch))
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f'Saved intermediate checkpoint: {epoch_ckpt_path}')

        if stop:  # 如果早停触发（stop=True），终止训练
            break

        # wlx：一个 epoch 完成，临时缓存可清理
        torch.cuda.empty_cache()  # 清理训练/验证中产生的缓存

    # 训练后评估与返回结果
    print('Done!')
    ckpt_path = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))  # 加载最佳模型权重（从早停保存的检查点）
    else:
        print(f'Checkpoint {ckpt_path} not found, using current model state.')
    _, val_cindex = summary_survival(model, val_loader, args.n_classes, chunk_size=args.chunk_size if hasattr(args,
                                                                                                              'chunk_size') else 10000)  # qw1215修改：添加chunk_size参数
    print('Val c-Index: {:.4f}'.format(val_cindex))
    if (not args.k_fold):  # 如果不是K折，额外计算测试集的C-index，返回结果字典和指标
        results_dict, test_cindex = summary_survival(model, test_loader, args.n_classes,
                                                     chunk_size=args.chunk_size if hasattr(args,
                                                                                           'chunk_size') else 10000)  # qw1215修改：添加chunk_size参数
        print('Test c-Index: {:.4f}'.format(test_cindex))
        if writer is not None:
            writer.close()
        return results_dict, test_cindex, val_cindex
    if writer is not None:
        writer.close()
    return val_cindex


'''
def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, scaler = None):
    # (1)显存调优,降低梯度累计逻辑，降低gc,改为gc=4
    # gc = 4

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")     # 检查是否有可用的CUDA设备，如果有则使用GPU，否则使用CPU。
    model.train()                               # 将模型设置为训练模式。
    train_loss_surv, train_loss = 0., 0.        # 初始化训练损失变量。

    print('\n')
    # 初始化数组:用于存储风险分数、审查状态和事件时间。
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):

        # data_WSI, data_omic, label, event_time, c ,path_coords = batch   # 解包批次数据
        data_WSI_1024, data_omic, label, event_time, c, path_coords_1024, data_WSI_128, path_coords_128 = batch  # 解包批次数据

        data_WSI = {
            'data_WSI_1024': data_WSI_1024,  # 大 patch 特征 (N, 1024)，可上 GPU
            'data_WSI_128': data_WSI_128.cpu()  # 小 patch 特征 (M, 512)，保留在 CPU！
        }
        data_WSI['data_WSI_1024'] = data_WSI['data_WSI_1024'].to(device, non_blocking=True)

        #data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        #data_WSI = data_WSI_1024.to(device, non_blocking=True)

        path_coords_1024 = path_coords_1024.to(device, non_blocking=True)
        path_coords_128 = path_coords_128.to(device, non_blocking=True)

        data_omic = data_omic.to(device, non_blocking=True)
        label = label.to(device, non_blocking = True)
        c = c.to(device, non_blocking=True)


        # wlx使用 autocast 上下文进行前向传播
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):     # 启用AMP        
            # hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
            hazards, S, Y_hat, _, _ = model(data_WSI)       # 通过模型前向传播，获取：hazards：风险函数;S：生存函数;Y_hat：预测输出;忽略其他两个返回值

            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)      # 计算损失值
            loss_value = loss.item()                                # 获取损失值的标量表示

            # 如果提供了正则化函数，则计算正则化损失，否则设为0。
            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
            # 计算风险分数，并将其从计算图中分离，转移到CPU并转换为NumPy数组。
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            # 存储当前批次的风险分数、审查状态和事件时间。
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            # 累积生存损失和总损失。
            train_loss_surv += loss_value
            train_loss += loss_value + loss_reg

            loss = loss / gc + loss_reg

        # 每100个批次打印一次训练进度信息。
        if (batch_idx + 1) % 50 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI['data_WSI_1024'].size(0)))

        # 反向传播：使用 scaler 缩放损失
        if scaler is not None:
            #wlx  print('AMP:反向传播：使用 scaler 缩放损失')
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 每 gc 步更新一次：使用 scaler.step 和 scaler.update
        if (batch_idx + 1) % gc == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    # 计算整个数据集的平均损失。
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # 计算一致性指数（C-index），用于评估模型性能。
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # 打印当前周期的训练结果。
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
'''


# qw1215开始：添加分块处理函数
def chunk_model_forward(model, data_WSI, chunk_size=10000, is_training=False):
    """
    对单一WSI进行分块处理，避免内存溢出
    保持模型输出与原始输出一致

    Args:
        model: 训练模型
        data_WSI: WSI数据，格式可能是dict或tensor
        chunk_size: 每次处理的patch数量
        is_training: 是否处于训练模式，如果是则需要保留梯度计算

    Returns:
        hazards, survival, Y_hat: 与原模型输出格式一致
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 处理不同格式的输入
    if isinstance(data_WSI, dict):
        # 如果是字典格式，需要处理每个键中的数据
        if 'patch_features' in data_WSI:
            patch_features = data_WSI['patch_features']
            if len(patch_features.shape) == 2:  # [N, D]
                num_patches = patch_features.shape[0]
            else:  # [B, N, D]
                num_patches = patch_features.shape[1]

            # 分块处理
            chunk_hazards = []
            chunk_survival = []
            chunk_Y_hat = []


            for i in range(0, num_patches, chunk_size):
                end_idx = min(i + chunk_size, num_patches)
                #print('{} : num_patches={}, chunk_size={}, end_idx={}'.format(i, num_patches, chunk_size, end_idx))

                if len(patch_features.shape) == 2:
                    chunk_data = {'patch_features': patch_features[i:end_idx]}
                else:
                    chunk_data = {'patch_features': patch_features[:, i:end_idx, :]}

                # 临时创建模型输入
                chunk_input = chunk_data
                if 'patch_coords' in data_WSI:
                    if len(data_WSI['patch_coords'].shape) == 2:
                        chunk_input['patch_coords'] = data_WSI['patch_coords'][i:end_idx]
                    else:
                        chunk_input['patch_coords'] = data_WSI['patch_coords'][:, i:end_idx, :]

                # qw1215修复：根据is_training决定是否使用no_grad
                if is_training:
                    # 训练模式下保留梯度计算
                    chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_input)
                else:
                    # 评估模式下使用no_grad节省内存
                    with torch.no_grad():
                        chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_input)

                chunk_hazards.append(chunk_hazards_i)
                chunk_survival.append(chunk_survival_i)
                chunk_Y_hat.append(chunk_Y_hat_i)

                # 清理内存
                del chunk_input
                torch.cuda.empty_cache()

            # 合并结果 - 这里采用平均的方式，可根据具体需求调整
            hazards = torch.mean(torch.stack(chunk_hazards), dim=0)
            survival = torch.mean(torch.stack(chunk_survival), dim=0)
            Y_hat = torch.mode(torch.cat(chunk_Y_hat, dim=0), dim=0)[0].unsqueeze(0)

        else:
            # 如果是多尺度数据，需要分别处理
            chunk_hazards = []
            chunk_survival = []
            chunk_Y_hat = []

            # 处理每个patch类型
            for patch_type in data_WSI:
                patch_data = data_WSI[patch_type]
                if len(patch_data.shape) == 2:  # [N, D]
                    num_patches = patch_data.shape[0]
                else:  # [B, N, D]
                    num_patches = patch_data.shape[1]

                for i in range(0, num_patches, chunk_size):
                    end_idx = min(i + chunk_size, num_patches)

                    if len(patch_data.shape) == 2:
                        chunk_data = {patch_type: patch_data[i:end_idx]}
                    else:
                        chunk_data = {patch_type: patch_data[:, i:end_idx, :]}

                    # qw1215修复：根据is_training决定是否使用no_grad
                    if is_training:
                        # 训练模式下保留梯度计算
                        chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_data)
                    else:
                        # 评估模式下使用no_grad节省内存
                        with torch.no_grad():
                            chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_data)

                    chunk_hazards.append(chunk_hazards_i)
                    chunk_survival.append(chunk_survival_i)
                    chunk_Y_hat.append(chunk_Y_hat_i)

                    # 清理内存
                    torch.cuda.empty_cache()

            # 合并结果
            hazards = torch.mean(torch.stack(chunk_hazards), dim=0)
            survival = torch.mean(torch.stack(chunk_survival), dim=0)
            Y_hat = torch.mode(torch.cat(chunk_Y_hat, dim=0), dim=0)[0].unsqueeze(0)

    else:  # tensor格式
        if len(data_WSI.shape) == 2:  # [N, D]
            num_patches = data_WSI.shape[0]
        else:  # [B, N, D]
            num_patches = data_WSI.shape[1]

        chunk_hazards = []
        chunk_survival = []
        chunk_Y_hat = []

        for i in range(0, num_patches, chunk_size):
            end_idx = min(i + chunk_size, num_patches)

            if len(data_WSI.shape) == 2:
                chunk_input = data_WSI[i:end_idx].unsqueeze(0)  # 添加batch维度
            else:
                chunk_input = data_WSI[:, i:end_idx, :]

            # qw1215修复：根据is_training决定是否使用no_grad
            if is_training:
                # 训练模式下保留梯度计算
                chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_input)
            else:
                # 评估模式下使用no_grad节省内存
                with torch.no_grad():
                    chunk_hazards_i, chunk_survival_i, chunk_Y_hat_i, _, _ = model(chunk_input)

            chunk_hazards.append(chunk_hazards_i)
            chunk_survival.append(chunk_survival_i)
            chunk_Y_hat.append(chunk_Y_hat_i)

            # 清理内存
            torch.cuda.empty_cache()

        # 合并结果
        hazards = torch.mean(torch.stack(chunk_hazards), dim=0)
        survival = torch.mean(torch.stack(chunk_survival), dim=0)
        Y_hat = torch.mode(torch.cat(chunk_Y_hat, dim=0), dim=0)[0].unsqueeze(0)

    return hazards, survival, Y_hat, None, None


# qw1215结束：添加分块处理函数

def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None,
                        lambda_reg=0., gc=16, scaler=None, chunk_size=10000,
                        cox_buffer=None, cox_aux_weight=0.0, cox_buffer_size=8):
    # (1)显存调优,降低梯度累计逻辑，降低gc,改为gc=4
    # gc = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的CUDA设备，如果有则使用GPU，否则使用CPU。
    model.train()  # 将模型设置为训练模式。
    train_loss_surv, train_loss = 0., 0.  # 初始化训练损失变量。

    print('\n')
    # 初始化数组:用于存储风险分数、审查状态和事件时间。
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, batch in enumerate(loader):

        # data_WSI, data_omic, label, event_time, c ,path_coords = batch   # 解包批次数据
        data_WSI, data_omic, label, event_time, c, path_coords = batch  # 解包批次数据
        if isinstance(data_WSI, list):
            data_WSI = data_WSI[0]  # 变成 {'patch_1024': ..., 'patch_512': ...}
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        elif isinstance(data_WSI, dict):
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        else:
            data_WSI = data_WSI.to(device, non_blocking=True)

        path_coords = path_coords.to(device, non_blocking=True)
        data_omic = data_omic.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)

        # wlx使用 autocast 上下文进行前向传播
        #with torch.cuda.amp.autocast(enabled=(scaler is not None)):  # 启用AMP
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            # qw1215修改：使用分块处理来避免内存溢出，设置is_training=True
            # hazards, S, Y_hat, _, _ = model(data_WSI)       # 通过模型前向传播，获取：hazards：风险函数;S：生存函数;Y_hat：预测输出;忽略其他两个返回值

            # v2/v3: 对 CCM_MIL 启用辅助损失（方向一致性 + mask 熵正则 + spread_reg）
            is_ccm = hasattr(model, 'stage2_gate') or hasattr(model, 'use_spread_reg')
            bag_size_for_chunk = 0
            if isinstance(data_WSI, dict) and 'patch_features' in data_WSI:
                bag_size_for_chunk = data_WSI['patch_features'].shape[0] if len(data_WSI['patch_features'].shape) == 2 else data_WSI['patch_features'].shape[1]
            elif isinstance(data_WSI, torch.Tensor):
                bag_size_for_chunk = data_WSI.shape[0] if len(data_WSI.shape) == 2 else data_WSI.shape[1]

            if is_ccm and bag_size_for_chunk <= chunk_size:
                out, aux = model(data_WSI, return_aux=True)
                hazards, S, Y_hat, _, _ = out
            else:
                hazards, S, Y_hat, _, _ = chunk_model_forward(model, data_WSI, chunk_size,
                                                              is_training=True)
                aux = None

            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)

            # wlx在计算 loss 前
            '''
            if S.dim() == 3:
                S = S.squeeze(1)  # [B, n_classes]
            if c.dim() == 1:
                c = c.unsqueeze(1)  # [B, 1]
            Y = Y.unsqueeze(1) if Y.dim() == 1 else Y
            c = c.unsqueeze(1) if c.dim() == 1 else c
            '''

            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)  # 计算损失值
            loss_value = loss.item()  # 获取损失值的标量表示

            # v2/v3/v4: CCM_MIL 辅助损失（方向一致性 + mask 熵正则 + spread_reg + Cox-PH ranking）
            loss_aux = 0.0
            if aux is not None:
                if 'dir_risks' in aux:
                    dir_risks = aux['dir_risks']  # (num_dirs, 1, n_classes)
                    loss_dir_consistency = torch.std(dir_risks, dim=0).mean()
                    loss_aux = loss_aux + 0.1 * loss_dir_consistency
                if 'mask_entropy' in aux:
                    loss_aux = loss_aux + 0.01 * aux['mask_entropy']
                if 'spread_reg' in aux:
                    loss_aux = loss_aux + 0.05 * aux['spread_reg']

                # V4: Cox-PH auxiliary loss via pairwise survival ranking
                if 'risk_pred' in aux and cox_buffer is not None and cox_aux_weight > 0:
                    rp = aux['risk_pred']  # current step, has grad
                    et = float(event_time)
                    cs = int(c.item())
                    loss_rank = 0.0
                    count = 0
                    for prev_rp_val, prev_et, prev_cs in cox_buffer:
                        # If current died earlier than previous, current should have higher risk
                        if cs == 0 and et < prev_et:
                            loss_rank += F.relu(prev_rp_val - rp + 1.0)
                            count += 1
                        # If previous died earlier than current, previous should have higher risk
                        if prev_cs == 0 and prev_et < et:
                            loss_rank += F.relu(rp - prev_rp_val + 1.0)
                            count += 1
                    if count > 0:
                        loss_aux = loss_aux + cox_aux_weight * (loss_rank / count)
                    cox_buffer.append((rp.detach().cpu().item(), et, cs))
                    if len(cox_buffer) > cox_buffer_size:
                        cox_buffer.pop(0)

                if loss_aux != 0.0:
                    loss = loss + loss_aux

            # 如果提供了正则化函数，则计算正则化损失，否则设为0。
            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg
            # 计算风险分数，并将其从计算图中分离，转移到CPU并转换为NumPy数组。
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            # 存储当前批次的风险分数、审查状态和事件时间。
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            # 累积生存损失和总损失。
            train_loss_surv += loss_value
            train_loss += loss_value + loss_reg
            if aux is not None and loss_aux != 0.0:
                train_loss += loss_aux.item()

            loss = loss / gc + loss_reg

        # 每100个批次打印一次训练进度信息。
        bag_size = 0

        if isinstance(data_WSI, dict):
            if 'data_patch_large' in data_WSI:
                bag_size = (data_WSI['data_patch_large'].shape[0], data_WSI['data_patch_small'].shape[0])
            elif 'patch_features' in data_WSI:
                bag_size = data_WSI['patch_features'].shape[0]

        else:
            bag_size = data_WSI.size(0)  # 兼容旧代码

        if (batch_idx + 1) % 50 == 0:
            aux_str = ''
            if aux is not None and loss_aux != 0.0:
                aux_str = ', aux: {:.4f}'.format(loss_aux.item())
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}{}'.format(batch_idx,
                                                                                                             loss_value + loss_reg,
                                                                                                             label.item(),
                                                                                                             float(
                                                                                                                 event_time),
                                                                                                             float(
                                                                                                                 risk),
                                                                                                             bag_size,
                                                                                                             aux_str))

        # 反向传播：使用 scaler 缩放损失
        if scaler is not None:
            # wlx  print('AMP:反向传播：使用 scaler 缩放损失')
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 每 gc 步更新一次：使用 scaler.step 和 scaler.update
        if (batch_idx + 1) % gc == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    # 计算整个数据集的平均损失。
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # 计算一致性指数（C-index），用于评估模型性能。
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships)
    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # 打印当前周期的训练结果。
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv,
                                                                                                 train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None,
                      loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, k_fold=False, chunk_size=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 将模型设置为评估模式（eval()），关闭 Dropout 和 BatchNorm 的随机性。
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    # for batch_idx, (data_WSI, data_omic, label, event_time, c, coords) in enumerate(loader):
    for batch_idx, batch in enumerate(loader):

        # data_WSI, data_omic, label, event_time, c ,path_coords = batch   # 解包批次数据
        data_WSI, data_omic, label, event_time, c, path_coords = batch  # 解包批次数据

        if isinstance(data_WSI, list):
            data_WSI = data_WSI[0]  # 变成 {'patch_1024': ..., 'patch_512': ...}
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        elif isinstance(data_WSI, dict):
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        else:
            data_WSI = data_WSI.to(device, non_blocking=True)
        # data_WSI = data_WSI.to(device, non_blocking=True)

        path_coords = path_coords.to(device, non_blocking=True)

        data_omic = data_omic.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            # qw1215修改：使用分块处理来避免内存溢出，设置is_training=False
            # hazards, S, Y_hat, _, _ = model(data_WSI) # return hazards, S, Y_hat, A_raw, results_dict
            hazards, S, Y_hat, _, _ = chunk_model_forward(model, data_WSI, chunk_size,
                                                          is_training=False)  # return hazards, S, Y_hat, A_raw, results_dict
            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print(
        'Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss,
                                                                                         c_index))
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        if k_fold:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        else:
            early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            # wlx：清理缓存
            torch.cuda.empty_cache()  # # 清理验证阶段缓存
            return True
    # wlx：清理缓存
    torch.cuda.empty_cache()  # ✅ 正常结束也清理
    return False


def summary_survival(model, loader, n_classes, chunk_size=1000):  # qw1215修改：添加chunk_size参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    # for batch_idx, (data_WSI, data_omic, label, event_time, c, coords) in enumerate(loader):
    #    data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
    #    label = label.to(device)
    #    coords = coords.to(device)
    for batch_idx, batch in enumerate(loader):
        # data_WSI, data_omic, label, event_time, c ,path_coords = batch   # 解包批次数据
        data_WSI, data_omic, label, event_time, c, path_coords = batch  # 解包批次数据

        if isinstance(data_WSI, list):
            data_WSI = data_WSI[0]  # 变成 {'patch_1024': ..., 'patch_512': ...}
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        elif isinstance(data_WSI, dict):
            data_WSI = {k: v.to(device, non_blocking=True) for k, v in data_WSI.items()}
        else:
            data_WSI = data_WSI.to(device, non_blocking=True)

        # data_WSI = data_WSI.to(device, non_blocking=True)
        path_coords = path_coords.to(device, non_blocking=True)

        data_omic = data_omic.to(device)
        label = label.to(device)
        c = c.to(device)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            # qw1215修改：使用分块处理来避免内存溢出，设置is_training=False
            # hazards, survival, Y_hat, _, _ = model(data_WSI)
            hazards, survival, Y_hat, _, _ = chunk_model_forward(model, data_WSI, chunk_size, is_training=False)
            # hazards = torch.sigmoid(hazards)
            # S = torch.cumprod(1 - hazards, dim=1)
        # risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        # risk = np.ndarray.item(-torch.sum(survival, dim=1).cpu().numpy())
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        # event_time = np.asscalar(event_time)
        event_time = event_time.item()
        # c = np.asscalar(c)
        c = np.ndarray.item(c.cpu().numpy())
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time, 'censorship': c}})

    c_index = \
    concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # wlx：清理缓存
    torch.cuda.empty_cache()  # ✅ 推理结束，清理缓存
    return patient_results, c_index
