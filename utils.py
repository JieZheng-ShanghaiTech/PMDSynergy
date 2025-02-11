import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

def prepare_device(num):
    if torch.cuda.is_available():
        logging.info(f"CUDA is supported. CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        logging.info(f"Using device: cuda:{num} {torch.cuda.get_device_name(num)}")
        return torch.device(f'cuda:{num}')
    else:
        logging.info("CUDA is not supported in this PyTorch build.")
        return torch.device('cpu')

# 定义数据集类
class DrugSynergyDataset(Dataset):
    def __init__(self, dataframe, omics_latent, smiles_dict, geminimol_dict, fingerprint_radius=3, fingerprint_size=2048):
        self.dataframe = dataframe
        self.omics_latent = omics_latent
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_size = fingerprint_size
        self.smiles_dict = smiles_dict
        self.geminimol_dict = geminimol_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 获取药物A和药物B的SMILES
        drugA = self.dataframe.iloc[idx]['DrugA']
        drugB = self.dataframe.iloc[idx]['DrugB']
        
        # 使用RDKit生成摩根指纹
        molA = Chem.MolFromSmiles(drugA)
        fingerprintA = AllChem.GetMorganFingerprintAsBitVect(molA, self.fingerprint_radius, nBits=self.fingerprint_size)
        fingerprintA = torch.tensor(list(fingerprintA), dtype=torch.float)  # 转换为张量
        molB = Chem.MolFromSmiles(drugB)
        fingerprintB = AllChem.GetMorganFingerprintAsBitVect(molB, self.fingerprint_radius, nBits=self.fingerprint_size)
        fingerprintB = torch.tensor(list(fingerprintB), dtype=torch.float)  # 转换为张量
        
        # 获取药物的协同分数
        synergy_score = self.dataframe.iloc[idx]['Loewe']
        
        # 获取Cell_ID对应的omics特征
        Cell_ID = self.dataframe.iloc[idx]['Cell_ID']
        omics_latent_vector = self.omics_latent[Cell_ID]

        # 将协同分数和omics向量转换为PyTorch张量
        synergy_score_tensor = torch.tensor(synergy_score, dtype=torch.float32)
        omics_latent_vector_tensor = torch.tensor(omics_latent_vector, dtype=torch.float32)
        
        # 将药物A和药物B的SMILES转换为BERT输入
        encoding_A = self.smiles_dict[drugA]
        encoding_B = self.smiles_dict[drugB]
        
        # 将smile向量转换为PyTorch张量
        encoding_A = torch.tensor(encoding_A, dtype=torch.float32)
        encoding_B = torch.tensor(encoding_B, dtype=torch.float32)

        encoding_A = encoding_A.view(-1)
        encoding_B = encoding_B.view(-1)

        # 将药物A和药物B的SMILES转换为GEMINI输入
        geminimol_A = self.geminimol_dict[drugA]
        geminimol_B = self.geminimol_dict[drugB]

        geminimol_A = torch.tensor(geminimol_A, dtype=torch.float32)
        geminimol_B = torch.tensor(geminimol_B, dtype=torch.float32)

        return {
            "omics_latent": omics_latent_vector_tensor,
            "Cell_ID": Cell_ID, 
            "label": synergy_score_tensor, 
            "smilesA": drugA, 
            "smilesB": drugB, 
            "fingerprintA": fingerprintA, 
            "fingerprintB": fingerprintB,
            "encodingA": encoding_A,
            "encodingB": encoding_B,
            "geminimolA": geminimol_A,
            "geminimolB": geminimol_B
        }

def load_data(train_csv, val_csv, test_csv, omics_latent, smiles_dict, geminimol_dict, fingerprint_size, batch_size):
    
    logging.info("Loading data begin")
    
    # 加载训练、验证和测试集数据
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)
    test_data = pd.read_csv(test_csv)

    # 创建对应的数据集实例
    train_dataset = DrugSynergyDataset(train_data, omics_latent, smiles_dict, geminimol_dict, fingerprint_size=fingerprint_size)
    val_dataset = DrugSynergyDataset(val_data, omics_latent, smiles_dict, geminimol_dict, fingerprint_size=fingerprint_size)
    test_dataset = DrugSynergyDataset(test_data, omics_latent, smiles_dict, geminimol_dict, fingerprint_size=fingerprint_size)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info("----end")

    return train_loader, val_loader, test_loader
