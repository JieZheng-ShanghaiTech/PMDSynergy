import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from utils import load_data, prepare_device
from model import CombinedModel, train, validate, test
import torch.optim as optim
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import csv

from rdkit import RDLogger

# 禁用所有非错误日志信息
RDLogger.DisableLog('rdApp.*')

# 创建可视化与结果保存的目录
def create_output_dir():
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 获取当前日期和时间
    output_dir = os.path.join('results', now)
    os.makedirs(output_dir, exist_ok=True)  # 创建目录
    return output_dir

def save_plot(metrics_df, output_dir):
    plt.figure()
    plt.plot(metrics_df['Epoch'], metrics_df['Train_MSE'], label='Train MSE')
    plt.plot(metrics_df['Epoch'], metrics_df['Validate_MSE'], label='Validate MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE over Epochs')
    plt.legend()

    # 保存图表到指定目录
    plt_path = os.path.join(output_dir, 'mse_over_epochs.png')
    plt.savefig(plt_path)
    logging.info(f"Saved MSE plot at {plt_path}")
    plt.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建输出文件夹
    output_dir = create_output_dir()
    
    # 初始化模型
    fingerprint_size = 2048
    batch_size = 32
    
    smiles_dict = np.load('data/ChemBERTa_smiles_embeddings.npy', allow_pickle=True).item()

    geminimol_dict = np.load('data/geminimol_latent_dict.npy', allow_pickle=True).item()

    cell_latent = np.load('data/cell_line_latent_values.npy', allow_pickle=True).item()
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data('data/train_data.csv', 'data/val_data.csv', 'data/test_data.csv', cell_latent, smiles_dict, geminimol_dict, fingerprint_size, batch_size=batch_size)
    
    # 初始化设备
    device = prepare_device(0)
    
    # 实例化模型
    model = CombinedModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 增加 L2 正则化
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    num_epochs = 500 # 150
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    turn = 1 # 4
    pre_val_loss, pre_val_mse, pre_val_rmse, pre_val_mae, pre_val_r2 = 0, 0, 0, 0, 0

    # 初始化CSV文件保存性能指标
    metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
    with open(metrics_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Validate_Loss', 'Train_MSE', 'Validate_MSE', 'Train_RMSE', 'Validate_RMSE', 'Train_MAE', 'Validate_MAE', 'Train_R2', 'Validate_R2'])

    # 训练和测试循环
    for epoch in range(num_epochs):
        train_loss, train_mse, train_rmse, train_mae, train_r2 = train(model, train_loader, optimizer, criterion, device)
        if epoch % turn == 0:
            val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(model, val_loader, criterion, device)
            pre_val_loss, pre_val_mse, pre_val_rmse, pre_val_mae, pre_val_r2 = val_loss, val_mse, val_rmse, val_mae, val_r2
        else:
            val_loss, val_mse, val_rmse, val_mae, val_r2 = pre_val_loss, pre_val_mse, pre_val_rmse, pre_val_mae, pre_val_r2

        # 保存每个epoch的结果到CSV文件
        with open(metrics_path, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, train_mse, val_mse, train_rmse, val_rmse, train_mae, val_mae, train_r2, val_r2])

        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validate Loss: {val_loss:.4f}')
        logging.info(f'Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R^2: {train_r2:.4f}')
        logging.info(f'Validate MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}')
        
        # 使用学习率调度器
        scheduler.step(val_loss)
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        if epoch % turn == 0:
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved best model at {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Early stopping at epoch {epoch + 1}')
                    break
                else :
                    logging.info(f'Patience counter = {patience_counter}')

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    test_loss, test_mse, test_rmse, test_mae, test_r2 = test(model, test_loader, criterion, device, save_path=os.path.join(output_dir, "test_predictions.csv"))
    logging.info(f"Final Test Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R^2: {test_r2:.4f}")

    # 可视化并保存图表
    metrics_df = pd.read_csv(metrics_path)
    save_plot(metrics_df, output_dir)

if __name__ == '__main__':
    main()
