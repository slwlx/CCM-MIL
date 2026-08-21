from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re
import random

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset
from transformers.models.auto.image_processing_auto import model_type

from utils.utils import generate_split, nth
from wlx_start import use_h5


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        '''
        case_id:代表病例的唯一标识符;
        slide_id:指病理切片的唯一标识符。
            slide_id与case_id关联，通过病例ID可查询到该患者对应的所有病理切片信息。

        censorship（生存分析中的删失状态）
        在生存分析中，censorship表示患者是否在研究结束时仍存活或失访：
            0：患者死亡（事件发生，如因癌症去世）。
            1：患者存活或失访（数据删失，如研究结束时患者仍存活或失去联系）。
            该字段是生存分析的关键变量，用于确定生存时间的计算方式。

        survival_months（生存时间，单位：月）
        指患者从确诊到死亡或研究结束的时间长度，以月为单位。例如，若患者生存时间为12个月，则记录为12。在生存分析中，survival_months与censorship结合使用，可绘制生存曲线并计算中位生存期等指标。
        '''

        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:         # 如果 shuffle=True，打乱数据顺序。
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        # 加载CSV文件：
        slide_data = pd.read_csv(csv_path, low_memory=False)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        #若无 case_id 列，则从索引提取前 12 字符作为患者 ID（TCGA ID 格式如 TCGA-XX-XXXX）
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        #pdb.set_trace()

        # 设置标签列为生存时间，默认 'survival_months'，并验证其存在。
        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        # 去重得到每个患者的唯一记录 → patients_df
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        # 取未删失患者（censorship = 0，即死亡）→ uncensored_df，用于分箱
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        # 使用分位数切分（qcut） 对未删失患者的生存时间分成 n_bins（默认是4） 个区间 → 得到边界 q_bins
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        # 然后扩展边界以防新数据越界。
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        # 再次使用 cut 将所有患者（包括删失）映射到这些 bin 中，生成离散标签 label，插入第3列。
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        # 患者与切片映射，创建字典 patient_dict，将患者 ID 映射到其对应的切片 ID 列表
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        # 保存该字典供后续使用
        self.patient_dict = patient_dict

        # 更新 slide_data 为去重后的患者数据，并添加 slide_id = case_id（模拟单切片）
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        # 创建复合标签字典：(bin_index, censorship) → 整数类ID
        # 例如：(0, 0) = 0, (0, 1) = 1, (1, 0) = 2, (1, 1) = 3...
        # 这表示：不同生存区间（默认4） + 是否删失（2） → 不同类则的8；
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        # 给每个患者赋最终的整数标签（考虑删失状态），同时保留原始 bin 标签为 disc_label
        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        # 保存分箱边界和总类别数（通常是 n_bins * 2）
        self.bins = q_bins
        self.num_classes=len(self.label_dict)

        # 构造患者级数据字典
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        # 重排列顺序（把 label 放前面），保存元数据列（前12列），准备分类 ID 列表
        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        ### 是否加载基因 signatures（如通路基因集合），用于 omic 模态建模
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./dataset_csv_sig/signatures.csv')
        else:
            self.signatures = None

        # 再次打印摘要（冗余，但可保留）
        if print_info:
            self.summarize()

    # 类别 ID 准备
    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    # 患者数据准备
    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    # 数据分箱静态方法
    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    # 数据集长度：返回数据集长度（根据 patient_strat 决定是患者级别还是切片级别）
    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    # 数据集摘要：打印数据集摘要信息（标签列、类别数、样本分布等）
    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    # 数据分割：get_split_from_df：从 DataFrame 中提取指定分割（训练/验证/测试）
    # 从 CSV 中读取的分割文件（train/val/test 列表）创建对应的 Generic_Split 实例
    def get_split_from_df(self, backbone, patch_size, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
            split.set_backbone(backbone)
            split.set_patch_size(patch_size)
        else:
            split = None
        
        return split

    # 数据分割:return_splits：返回数据分割（支持从 ID 或 CSV 文件加载）
    def return_splits(self, backbone, patch_size = '', from_id: bool=True, csv_path: str=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, mode = self.mode, metadata= self.apply_sig, data_dir=self.data_dir, num_classes=self.num_classes, patient_dict=self.patient_dict, label_col=self.label_col)
                train_split.set_backbone(backbone)
                train_split.set_patch_size(patch_size)
                #print('hhhhhhhhhhhhhhhhhhhhhhhhh')
            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes, patient_dict=self.patient_dict, label_col=self.label_col)
                val_split.set_backbone(backbone)
                val_split.set_patch_size(patch_size)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, metadata = self.apply_sig, mode = self.mode, data_dir=self.data_dir, num_classes=self.num_classes, patient_dict=self.patient_dict, label_col=self.label_col)
                test_split.set_backbone(backbone)
                test_split.set_patch_size(patch_size)

            else:
                test_split = None
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(backbone, patch_size, all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            # print("****** Normalizing Data ******")
            # scalers = train_split.get_scaler()
            # train_split.apply_scaler(scalers=scalers)
            # val_split.apply_scaler(scalers=scalers)
            # test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split, test_split
    
    '''
    Added function create_splits from Generic_WSI_Classification_Dataset
    '''
    # 数据分割生成与设置:create_splits：生成数据分割配置。
    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
                    'n_splits' : k, 
                    'val_num' : val_num, 
                    'test_num': test_num,
                    'label_frac': label_frac,
                    'seed': self.seed,
                    'custom_test_ids': custom_test_ids
                    }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)
    
    
    '''
    Added function set_splits from Generic_WSI_Classification_Dataset
    '''
    # 数据分割生成与设置:set_splits：设置训练/验证/测试集的 ID
    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))] 

            for split in range(len(ids)): 
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None
    
    '''
    Added functions test_split_gen and save_split from Generic_WSI_Classification_Dataset
    '''

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                            columns= columns)
        df = df.reset_index(drop=True)
        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]
        
        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1) 
        df.to_csv(filename, index = False)

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        self.use_h5 = kwargs.pop('use_h5', False)
        #self.data_mode = kwargs.pop('data_mode', False)

        # 调用父类构造函数，加载 slide_data、genomic_features、patient_dict 等
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir    # 存放预提取特征的目录（.pt 或 .h5）
        self.mode = mode            # 数据模式：'path', 'omic', 'pathomic', 'coattn' 等

    # 切换是否使用 HDF5 格式加载数据
    def load_from_h5(self, toggle):
        toggle = True
        self.use_h5 = toggle        # toggle 为布尔值，True 表示启用 HDF5

    # ============================================================
    # wlx添加
    def build_patch_mapping(self, csv_path):
        """
        读取 CSV，返回 dict: big_idx -> list of small_idx
        """
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            big_idx = row['big_patch_index']
            small_idx = row['small_patch_index']
            if big_idx not in mapping:
                mapping[big_idx] = []
            mapping[big_idx].append(small_idx)
        return mapping  # {0: [0,1,2], 1: [3,4], ...}
    # =============================================================

    # 核心方法：根据索引 idx 返回一个训练样本
    # inx 通过 DataLoader 来自动遍历数据集
    def __getitem__(self, idx):
        # slide_data 是在 父类 Generic_WSI_Survival_Dataset.__init__() 中加载的
        # 从 slide_data 中获取当前样本的患者 ID、标签、生存时间和删失状态
        case_id = self.slide_data['case_id'][idx]           # 患者标识（根据csv文件）
        label = self.slide_data['disc_label'][idx]          # 离散化生存标签（discretized survival label），即生存风险类别(由父类生成）
                                                            # 例如：0=短期死亡风险（<1年），1=1-3年，2=3-5年，3=长期存活
        event_time = self.slide_data[self.label_col][idx]   # 患者的实际生存时间
        c = self.slide_data['censorship'][idx]              # 删失状态（censorship status）

        # 获取该患者对应的所有 WSI 切片 ID（一个患者可能有多个 WSI）
        slide_ids = self.patient_dict[case_id]              # 一个患者可能有多个 WSI 切片（例如：原发灶、转移灶、不同组织块）
        #print('slide_ids: {}'.format(slide_ids))

        # 如果 data_dir 是字典，则根据癌症类型（oncotree_code）选择对应路径
        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir        # 否则使用统一路径

        '''
        ===============================================================
        wlx 修改               
        '''
        if self.mode == 'ms-path':
            data_dir_1024_h5 = os.path.join(data_dir, 'dim1024/h5_files')
            data_dir_128_h5 = os.path.join(data_dir, 'dim128/h5_files')
            map_dir = os.path.join(data_dir, 'dim128/patches')
        '''
        ===============================================================
        wlx 修改内容              
        '''


        # 当前未使用 HDF5，且提供了 data_dir
        if not self.use_h5:
            if self.data_dir:
                # 模式 1: 仅使用病理图像特征（patch-level）
                if self.mode == 'path':
                    #print(idx,'---------------dataset_survival.py: 406,path-------------------------')
                    path_features = []
                    for slide_id in slide_ids:
                        # wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        # 构造 .pt 特征文件路径（去掉 .svs 后缀）
                        # '{}.pt'.format(slide_id.rstrip('.svs'))将从csv文件读取的slide_id的WSI切片类型.svs替换成了.pt
                        wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    # 将该患者所有 WSI 的特征拼接成一个 bag
                    path_features = torch.cat(path_features, dim=0)

                    '''
                    ********************修改程序的地方2：wsi  bag_size过大的path_features随机丢弃***************************
                    '''
                    #print('path_features.size(0)=', path_features.size(0))
                    max_patch_num = 100000
                    if path_features.size(0) > max_patch_num:
                        #print()
                        #print('slide_ids too big: {}'.format(slide_ids),'path_features.size(0)=', path_features.size(0))
                        #print('path_features.size(0)=', path_features.size(0))
                        num_drop = path_features.size(0) - max_patch_num
                        # 随机选择要丢弃的 patch 索引
                        drop_indices = torch.randperm(path_features.size(0))[:num_drop]
                        mask = torch.ones(path_features.size(0), dtype=torch.bool)
                        mask[drop_indices] = False
                        path_features = path_features[mask]
                        #print('new_path_features.size(0)=', path_features.size(0))
                        #print()

                    # 返回图像特征 + 空占位符（因无 omic）+ 标签信息
                    # 模型解包：data_WSI, data_omic, label, event_time, c = batch   # 解包批次数据
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

                elif self.mode == 'cluster':
                    #print('---------------dataset_survival.py: 417,cluster-------------------------')
                    #input('press enter to continue')
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt']) #! no fname2ids?
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    print('---------------dataset_survival.py: 431 omic-------------------------')
                    input('press enter to continue')
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    print('---------------dataset_survival.py: 436,pathomic-------------------------')
                    input('press enter to continue')
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'coattn':
                    print('---------------dataset_survival.py: 447,coattn-------------------------')
                    input('press enter to continue')
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c
        else:
            if self.mode == 'ss-path':
                path_features = []
                path_coords = []  # 新增：用于存储坐标
                #data_dir = os.path.join(data_dir, 'dim1024_1024_1024/h5_files')

                for slide_id in slide_ids:
                    # 构造 .h5 文件路径（去掉 .svs 后缀）
                    h5_file_name = '{}.h5'.format(slide_id.rstrip('.svs'))
                    h5_path = os.path.join(data_dir, h5_file_name)

                    if not os.path.exists(h5_path):
                        raise FileNotFoundError(f"H5 file not found: {h5_path}")

                    # 从 .h5 文件加载 features 和 coords
                    with h5py.File(h5_path, 'r') as f:
                        # 加载特征
                        if 'features' not in f:
                            raise KeyError(f"'features' dataset not found in {h5_path}")
                        feats = f['features'][:]  # numpy array
                        feats = torch.from_numpy(feats).float()  # -> Tensor

                        # 加载坐标
                        if 'coords' not in f:
                            raise KeyError(f"'coords' dataset not found in {h5_path}")
                        coords = f['coords'][:]  # numpy array
                        coords = torch.from_numpy(coords).long()  # 通常是整数坐标

                    # 将当前 slide 的 features 和 coords 加入列表
                    path_features.append(feats)
                    path_coords.append(coords)

                # 拼接所有 slide 的 features 和 coords
                path_features = torch.cat(path_features, dim=0)  # [N, D]
                path_coords = torch.cat(path_coords, dim=0)  # [N, 2] 或 [N, 3]


                ##********************修改程序的地方2：wsi bag_size过大的path_features随机丢弃***************************
                '''
                max_patch_num = 70000
                if path_features.size(0) > max_patch_num:
                    num_keep = max_patch_num
                    # 随机采样保留的 indices
                    keep_indices = torch.randperm(path_features.size(0))[:num_keep]

                    # 只保留随机采样的 patch
                    path_features = path_features[keep_indices]
                    path_coords = path_coords[keep_indices]
                '''
                #print("不在进行max_patch_num={}判断及随机采用")

                #print(path_features.shape,path_coords.shape)
                path_features = {
                    'patch_features': path_features,
                    'patch_coords': path_coords
                }



                # ✅ 返回：features, coords, dummy omic, label, event_time, c
                return (path_features, torch.zeros((1, 1)), label, event_time, c, path_coords)

            elif self.mode == 'ms-path':
                # print('---------------ms-path---------------')

                path_coords = []  # 新增：用于存储坐标

                path_features_large = []
                path_coords_large = []  # 新增：用于存储坐标

                path_features_small = []
                path_coords_small = []  # 新增：用于存储坐标

                dfs_postorder_index_large = []
                dfs_preorder_index_large = []
                bfs_levelorder_index_large = []
                wsi_3gride_index_large = []
                wsi_7gride_index_large = []

                dfs_postorder_index_small = []
                dfs_preorder_index_small = []
                bfs_levelorder_index_small = []
                wsi_3gride_index_small = []
                wsi_7gride_index_small = []

                for slide_id in slide_ids:
                    # 构造 .h5 文件路径（去掉 .svs 后缀）
                    data_dir_large = '/home/wlx/github/MambaMIL-main/dataset/BLCA/pt_files/resnet50/dim1024_1024_1024/h5_files'
                    data_dir_small = '/home/wlx/github/MambaMIL-main/dataset/BLCA/pt_files/resnet50/dim512_512/h5_files'

                    h5_file_name = '{}.h5'.format(slide_id.rstrip('.svs'))
                    h5_path_large = os.path.join(data_dir_large, h5_file_name)
                    h5_path_small = os.path.join(data_dir_small, h5_file_name)
                    map_file_name = '{}.csv'.format(slide_id.rstrip('.svs'))
                    map_file = os.path.join(map_dir, map_file_name)

                    if not os.path.exists(h5_path_large):
                        raise FileNotFoundError(f"H5 file not found: {h5_path_large}")
                    elif not os.path.exists(h5_path_small):
                        raise FileNotFoundError(f"H5 file not found: {h5_path_small}")
                    # 从 .h5 文件加载 features 和 coords
                    with h5py.File(h5_path_large, 'r') as f:
                        # 加载特征
                        if 'features' not in f:
                            raise KeyError(f"'features' dataset not found in {h5_path_large}")
                        features_1024 = f['features'][:]  # numpy array
                        features_1024 = torch.from_numpy(features_1024).float()  # -> Tensor

                        # 加载坐标
                        if 'coords' not in f:
                            raise KeyError(f"'coords' dataset not found in {h5_path_large}")
                        coords_1024 = f['coords'][:]  # numpy array
                        coords_1024 = torch.from_numpy(coords_1024).long()  # 通常是整数坐标

                        # 加载深度搜索（反向）
                        if 'dfs_postorder_index' not in f:
                            raise KeyError(f"'dfs_postorder_index' dataset not found in {h5_path_large}")
                        dfs_postorder_index_1024 = f['dfs_postorder_index'][:]  # numpy array
                        dfs_postorder_index_1024 = torch.from_numpy(dfs_postorder_index_1024).long()

                        # 加载深度搜索（反向）
                        if 'dfs_preorder_index' not in f:
                            raise KeyError(f"'dfs_preorder_index' dataset not found in {h5_path_large}")
                        dfs_preorder_index_1024 = f['dfs_preorder_index'][:]  # numpy array
                        dfs_preorder_index_1024 = torch.from_numpy(dfs_preorder_index_1024).long()

                        # 加载广度搜索
                        if 'bfs_levelorder_index' not in f:
                            raise KeyError(f"'bfs_levelorder_index' dataset not found in {h5_path_large}")
                        bfs_levelorder_index_1024 = f['bfs_levelorder_index'][:]  # numpy array
                        bfs_levelorder_index_1024 = torch.from_numpy(bfs_levelorder_index_1024).long()

                        # 加载wsi局部3*3窗口扫描搜索
                        if 'wsi_3gride_index' not in f:
                            raise KeyError(f"'wsi_3gride_index' dataset not found in {h5_path_large}")
                        wsi_3gride_index_1024 = f['wsi_3gride_index'][:]  # numpy array
                        wsi_3gride_index_1024 = torch.from_numpy(wsi_3gride_index_1024).long()

                        # 加载wsi局部7*7窗口扫描搜索
                        if 'wsi_7gride_index' not in f:
                            raise KeyError(f"'wsi_7gride_index' dataset not found in {h5_path_large}")
                        wsi_7gride_index_1024 = f['wsi_7gride_index'][:]  # numpy array
                        wsi_7gride_index_1024 = torch.from_numpy(wsi_7gride_index_1024).long()

                    # 将当前 slide 的 features 和 coords 加入列表
                    path_features_large.append(features_1024)
                    path_coords_large.append(coords_1024)
                    dfs_postorder_index_large.append(dfs_postorder_index_1024)
                    dfs_preorder_index_large.append(dfs_preorder_index_1024)
                    bfs_levelorder_index_large.append(bfs_levelorder_index_1024)
                    wsi_3gride_index_large.append(wsi_3gride_index_1024)
                    wsi_7gride_index_large.append(wsi_7gride_index_1024)

                    with h5py.File(h5_path_small, 'r') as f:
                        # 加载特征
                        if 'features' not in f:
                            raise KeyError(f"'features' dataset not found in {h5_path_small}")
                        features_128 = f['features'][:]  # numpy array
                        features_128 = torch.from_numpy(features_128).float()  # -> Tensor

                        # 加载坐标
                        if 'coords' not in f:
                            raise KeyError(f"'coords' dataset not found in {h5_path_small}")
                        coords_128 = f['coords'][:]  # numpy array
                        coords_128 = torch.from_numpy(coords_128).long()  # 通常是整数坐标

                        # 加载深度搜索（反向）
                        if 'dfs_postorder_index' not in f:
                            raise KeyError(f"'dfs_postorder_index' dataset not found in {h5_path_small}")
                        dfs_postorder_index_128 = f['dfs_postorder_index'][:]  # numpy array
                        dfs_postorder_index_128 = torch.from_numpy(dfs_postorder_index_128).long()

                        # 加载深度搜索（反向）
                        if 'dfs_preorder_index' not in f:
                            raise KeyError(f"'dfs_preorder_index' dataset not found in {h5_path_small}")
                        dfs_preorder_index_128 = f['dfs_preorder_index'][:]  # numpy array
                        dfs_preorder_index_128 = torch.from_numpy(dfs_preorder_index_128).long()

                        # 加载广度搜索
                        if 'bfs_levelorder_index' not in f:
                            raise KeyError(f"'bfs_levelorder_index' dataset not found in {h5_path_small}")
                        bfs_levelorder_index_128 = f['bfs_levelorder_index'][:]  # numpy array
                        bfs_levelorder_index_128 = torch.from_numpy(bfs_levelorder_index_128).long()

                        # 加载wsi局部3*3窗口扫描搜索
                        if 'wsi_3gride_index' not in f:
                            raise KeyError(f"'wsi_3gride_index' dataset not found in {h5_path_large}")
                        wsi_3gride_index_128 = f['wsi_3gride_index'][:]  # numpy array
                        wsi_3gride_index_128 = torch.from_numpy(wsi_3gride_index_128).long()

                        # 加载wsi局部7*7窗口扫描搜索
                        if 'wsi_7gride_index' not in f:
                            raise KeyError(f"'wsi_7gride_index' dataset not found in {h5_path_large}")
                        wsi_7gride_index_128 = f['wsi_7gride_index'][:]  # numpy array
                        wsi_7gride_index_128 = torch.from_numpy(wsi_7gride_index_128).long()

                    # 将当前 slide 的 features 和 coords 加入列表
                    path_features_small.append(features_128)
                    path_coords_small.append(coords_128)
                    dfs_postorder_index_small.append(dfs_postorder_index_128)
                    dfs_preorder_index_small.append(dfs_preorder_index_128)
                    bfs_levelorder_index_small.append(bfs_levelorder_index_128)
                    wsi_3gride_index_small.append(wsi_3gride_index_128)
                    wsi_7gride_index_small.append(wsi_7gride_index_128)


                # 拼接所有 slide 的 features 和 coords
                path_features_large = torch.cat(path_features_large, dim=0)  # [N, D]
                path_coords_large = torch.cat(path_coords_large, dim=0)  # [N, 2] 或 [N, 3]
                dfs_postorder_index_large = torch.cat(dfs_postorder_index_large, dim=0)
                dfs_preorder_index_large = torch.cat(dfs_preorder_index_large, dim=0)
                bfs_levelorder_index_large = torch.cat(bfs_levelorder_index_large, dim=0)
                wsi_3gride_index_large = torch.cat(wsi_3gride_index_large, dim=0)
                wsi_7gride_index_large = torch.cat(wsi_7gride_index_large, dim=0)

                path_features_small = torch.cat(path_features_small, dim=0)  # [N, D]
                path_coords_small = torch.cat(path_coords_small, dim=0)  # [N, 2] 或 [N, 3]
                dfs_postorder_index_small = torch.cat(dfs_postorder_index_small, dim=0)
                dfs_preorder_index_small = torch.cat(dfs_preorder_index_small, dim=0)
                bfs_levelorder_index_small = torch.cat(bfs_levelorder_index_small, dim=0)
                wsi_3gride_index_small = torch.cat(wsi_3gride_index_small, dim=0)
                wsi_7gride_index_small = torch.cat(wsi_7gride_index_small, dim=0)

                '''
                ********************修改程序的地方2：wsi bag_size过大的path_features随机丢弃***************************
                
                max_patch_num = 100000
                if path_features_large.size(0) > max_patch_num:
                    num_keep = max_patch_num
                    # 随机采样保留的 indices
                    keep_indices = torch.randperm(path_features_large.size(0))[:num_keep]

                    # 只保留随机采样的 patch
                    path_features_large = path_features_large[keep_indices]
                    path_coords_large = path_coords_large[keep_indices]
                elif path_features_small.size(0) > max_patch_num:
                    num_keep = max_patch_num
                    keep_indices = torch.randperm(path_features_small.size(0))[:num_keep]
                    path_features_small = path_features_small[keep_indices]
                    path_coords_small = path_coords_small[keep_indices]
                '''
                #print("不在进行max_patch_num={}判断及随机采用（ms）")

                path_features = {
                    'data_patch_large': path_features_large,
                    'coords_patch_large': path_coords_large,
                    'dfs_postorder_index_large': dfs_postorder_index_large,
                    'dfs_preorder_index_large': dfs_preorder_index_large,
                    'bfs_levelorder_index_large': bfs_levelorder_index_large,
                    'coords_patch_small': path_coords_small,
                    'data_patch_small': path_features_small,
                    'dfs_postorder_index_small': dfs_postorder_index_small,
                    'dfs_preorder_index_small': dfs_preorder_index_small,
                    'bfs_levelorder_index_small': bfs_levelorder_index_small,
                    'wsi_3gride_index_large': wsi_3gride_index_large,
                    'wsi_7gride_index_large': wsi_7gride_index_large,
                    'wsi_3gride_index_small': wsi_3gride_index_small,
                    'wsi_7gride_index_small': wsi_7gride_index_small
                }


                return (path_features, torch.zeros((1, 1)), label, event_time, c, path_coords_small)




class Generic_Split(Generic_MIL_Survival_Dataset):
    # Generic_Split 类继承自 Generic_MIL_Survival_Dataset，主要用于数据分割和预处理。
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = use_h5
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        #! add from HIPT
        # cluster_dir = "/".join(data_dir.split("/")[0:-1])
        # if os.path.isfile(os.path.join(cluster_dir, 'fast_cluster_ids.pkl')):
        #     with open(os.path.join(cluster_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
        #         self.fname2ids = pickle.load(handle)
        # else:
        #     print("Cluster file missing")
        ### --> Initializing genomic features in Generic Split
        # self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        # self.signatures = signatures

        # with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
        #     self.fname2ids = pickle.load(handle)

        # def series_intersection(s1, s2):
        #     return pd.Series(list(set(s1) & set(s2)))

        # if self.signatures is not None:
        #     self.omic_names = []
        #     for col in self.signatures.columns:
        #         omic = self.signatures[col].dropna().unique()
        #         omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
        #         omic = sorted(series_intersection(omic, self.genomic_features.columns))
        #         self.omic_names.append(omic)
        #     self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--

    def set_backbone(self, backbone):
        print('Setting Backbone:', backbone)
        self.backbone = backbone

    def set_patch_size(self, size):
        print('Setting Patchsize:', size)
        self.patch_size = size

    def pre_loading(self, thread=8):
        # set flag
        self.cache_flag = True

        ids = list(range(len(self)))
        from multiprocessing.pool import ThreadPool
        exe = ThreadPool(thread)
        exe.map(self.__getitem__, ids)