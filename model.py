import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import logging
import time
import pandas as pd
import numpy as np
import re
import math

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 增加 Dropout 比例

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
class BilinearFusion(nn.Module):
    def __init__(self, A_dim, B_dim, fusion_dim):
        super(BilinearFusion, self).__init__()
        self.bilinear = nn.Bilinear(A_dim, B_dim, fusion_dim)

    def forward(self, A, B):
        return self.bilinear(A, B)

class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 512)  # 用于计算注意力权重 128
        self.fc2 = nn.Linear(512, 1)  # 输出一个标量，表示该模态的注意力权重
        self.softmax = nn.Softmax(dim=1)  # 对注意力权重进行归一化
        
        # 定义线性变换，映射输入到Q、K、V空间
        self.query_fc = nn.Linear(input_size, input_size)
        self.key_fc = nn.Linear(input_size, input_size)
        self.value_fc = nn.Linear(input_size, input_size)

    def forward(self, x1, x2, x3):
        # 将输入向量应用线性变换得到Q、K、V
        Q1 = self.query_fc(x1)  # [batch_size, input_size]
        K1 = self.key_fc(x1)    # [batch_size, input_size]
        V1 = self.value_fc(x1)  # [batch_size, input_size]
        
        Q2 = self.query_fc(x2)  # [batch_size, input_size]
        K2 = self.key_fc(x2)    # [batch_size, input_size]
        V2 = self.value_fc(x2)  # [batch_size, input_size]
        
        Q3 = self.query_fc(x3)  # [batch_size, input_size]
        K3 = self.key_fc(x3)    # [batch_size, input_size]
        V3 = self.value_fc(x3)  # [batch_size, input_size]

        # 计算注意力权重并加权值
        # 对每一组（Q1, K2, V2）进行自注意力计算
        attn_weights1 = torch.matmul(Q1, K2.transpose(1, 0)) / (self.input_size ** 0.5)  # [batch_size, batch_size]
        attn_weights1 = F.softmax(attn_weights1, dim=-1)  # [batch_size, batch_size]
        output1 = torch.matmul(attn_weights1, V2)  # [batch_size, input_size]
        
        # 对每一组（Q2, K3, V3）进行自注意力计算
        attn_weights2 = torch.matmul(Q2, K3.transpose(1, 0)) / (self.input_size ** 0.5)  # [batch_size, batch_size]
        attn_weights2 = F.softmax(attn_weights2, dim=-1)  # [batch_size, batch_size]
        output2 = torch.matmul(attn_weights2, V3)  # [batch_size, input_size]
        
        # 对每一组（Q3, K1, V1）进行自注意力计算
        attn_weights3 = torch.matmul(Q3, K1.transpose(1, 0)) / (self.input_size ** 0.5)  # [batch_size, batch_size]
        attn_weights3 = F.softmax(attn_weights3, dim=-1)  # [batch_size, batch_size]
        output3 = torch.matmul(attn_weights3, V1)  # [batch_size, input_size]

        # 计算每个模态的注意力权重
        attn1 = self.fc2(F.relu(self.fc1(output1))).squeeze(-1)  # [batchsize]
        attn2 = self.fc2(F.relu(self.fc1(output2))).squeeze(-1)  # [batchsize]
        attn3 = self.fc2(F.relu(self.fc1(output3))).squeeze(-1)  # [batchsize]

        # 拼接注意力权重，进行softmax归一化
        attn_weights = torch.stack([attn1, attn2, attn3], dim=1)  # [batchsize, 3]
        attn_weights = self.softmax(attn_weights)  # [batchsize, 3]

        # 根据注意力权重加权三个模态的特征
        output = attn_weights[:, 0].unsqueeze(-1) * x1 + attn_weights[:, 1].unsqueeze(-1) * x2 + attn_weights[:, 2].unsqueeze(-1) * x3
        
        # 返回三个加权后的输出向量
        return output
    
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_A, emb_B, S_pred):
        """
        :param emb_A: 药物A的嵌入
        :param emb_B: 药物B的嵌入
        :param S_pred: 药物对的协同分数
        """
        # 计算药物 A 和药物 B 之间的余弦相似度
        sim_A_B = F.cosine_similarity(emb_A, emb_B) / self.temperature
        
        # 初始化对比损失
        positive_loss = 0
        negative_loss = 0

        # 处理正样本（S_pred > 0，表示协同作用）
        pos_mask = S_pred > 0  # 协同作用强的样本对
        neg_mask = S_pred < 0  # 拮抗作用强的样本对

        # 对于正样本，应该最大化相似度（即让它们更接近）
        positive_loss = torch.sum(torch.log(torch.exp(sim_A_B) / (torch.exp(sim_A_B) + 1)) * pos_mask)

        # 对于负样本，应该最小化相似度（即让它们更远离）
        negative_loss = torch.sum(torch.log(1 / (torch.exp(sim_A_B) + 1)) * neg_mask)

        # 对比损失：将正样本和负样本的损失结合，平衡它们的影响
        loss = (positive_loss + negative_loss) / torch.sum(pos_mask + neg_mask)

        return loss

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.linear = nn.Linear(768, 2048)
        self.attention = AttentionModule(input_size=2048)
        self.fusion = BilinearFusion(A_dim=2048, B_dim=200, fusion_dim=128)
        self.contrastive_loss = NTXentLoss()
        self.mlp = MLP(256)

    def forward(self, encoding_A, encoding_B, geminimolA, geminimolB, fingerprintA, fingerprintB, omics_latent_vectors, labels):
        da = self.linear(encoding_A)
        db = self.linear(encoding_B)

        mA = self.attention(da, fingerprintA, geminimolA)
        mB = self.attention(db, fingerprintB, geminimolB)

        final_Ainput = self.fusion(mA, omics_latent_vectors)
        final_Binput = self.fusion(mB, omics_latent_vectors)

        # 计算对比损失
        contrastive_loss = self.contrastive_loss(final_Ainput, final_Binput, labels)

        # 通过MLP进行预测
        output = self.mlp(torch.cat((final_Ainput, final_Binput), dim=1))  # 输出 [batch_size, 1]
        return output, contrastive_loss

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar):
        
        # 从字典中获取图和标签
        encoding_A = batch_data['encodingA'].to(device)
        encoding_B = batch_data['encodingB'].to(device)
        geminimolA = batch_data['geminimolA'].to(device)
        geminimolB = batch_data['geminimolB'].to(device)
        fingerprintA = batch_data['fingerprintA'].to(device)
        fingerprintB = batch_data['fingerprintB'].to(device)
        labels = batch_data['label'].to(device)
        omics_latent_vectors = batch_data['omics_latent'].to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs, contrastive_loss = model(encoding_A, encoding_B, geminimolA, geminimolB, fingerprintA, fingerprintB, omics_latent_vectors, labels)

        # 计算损失
        loss = criterion(outputs.squeeze(-1), labels)  # 确保输出形状与标签一致
        running_loss += loss.item()

        loss += 0.01 * contrastive_loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 收集预测值和真实值
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.squeeze(-1).detach().cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return avg_loss, mse, rmse, mae, r2

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):

            # 从字典中获取图和标签
            encoding_A = batch_data['encodingA'].to(device)
            encoding_B = batch_data['encodingB'].to(device)
            geminimolA = batch_data['geminimolA'].to(device)
            geminimolB = batch_data['geminimolB'].to(device)
            fingerprintA = batch_data['fingerprintA'].to(device)
            fingerprintB = batch_data['fingerprintB'].to(device)
            labels = batch_data['label'].to(device)
            omics_latent_vectors = batch_data['omics_latent'].to(device)

            # 前向传播
            outputs, contrastive_loss = model(encoding_A, encoding_B, geminimolA, geminimolB, fingerprintA, fingerprintB, omics_latent_vectors, labels)

            # 计算损失
            loss = criterion(outputs.squeeze(-1), labels)  # 确保输出形状与标签一致
            running_loss += loss.item()

            # 收集预测值和真实值
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.squeeze(-1).detach().cpu().numpy())

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(val_loader)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return avg_loss, mse, rmse, mae, r2

def test(model, test_loader, criterion, device, save_path="predictions.csv"):
    model.eval()  # 将模型设置为评估模式
    test_loss = 0.0
    y_true = []
    y_pred = []
    drugA_smiles = []  # 保存药物A的SMILES序列
    drugB_smiles = []  # 保存药物B的SMILES序列
    S_id = []

    progress_bar = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            # 获取药物A和药物B的SMILES序列
            drugA_smiles_batch = [test_loader.dataset[i]['smilesA'] for i in range(batch_idx * len(batch_data['label']), (batch_idx + 1) * len(batch_data['label']))]
            drugB_smiles_batch = [test_loader.dataset[i]['smilesB'] for i in range(batch_idx * len(batch_data['label']), (batch_idx + 1) * len(batch_data['label']))]

            # 从字典中获取图和标签
            encoding_A = batch_data['encodingA'].to(device)
            encoding_B = batch_data['encodingB'].to(device)
            geminimolA = batch_data['geminimolA'].to(device)
            geminimolB = batch_data['geminimolB'].to(device)
            fingerprintA = batch_data['fingerprintA'].to(device)
            fingerprintB = batch_data['fingerprintB'].to(device)
            Cell_id = batch_data['Cell_id']
            S_id.extend(Cell_id)
            labels = batch_data['label'].to(device)
            omics_latent_vectors = batch_data['omics_latent'].to(device)

            # 前向传播
            outputs, contrastive_loss = model(encoding_A, encoding_B, geminimolA, geminimolB, fingerprintA, fingerprintB, omics_latent_vectors, labels)

            # 计算损失
            loss = criterion(outputs.squeeze(-1), labels)  # 确保输出形状与标签一致
            test_loss += loss.item()

            # 更新进度条的损失
            progress_bar.set_postfix(loss=loss.item())

            # 收集真实值和预测值
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.squeeze(-1).detach().cpu().numpy())

            # 保存药物A和药物B的SMILES序列
            drugA_smiles.extend(drugA_smiles_batch)
            drugB_smiles.extend(drugB_smiles_batch)

    # 将结果保存为CSV
    results_df = pd.DataFrame({
        "DrugA_SMILES": drugA_smiles,
        "DrugB_SMILES": drugB_smiles,
        "Cell_id": S_id,
        "True_Label": y_true,
        "Predicted_Label": y_pred
    })
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    # 计算评价指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, mse, rmse, mae, r2