import os, sys, random, copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional, Union
from functools import partial, wraps
from tqdm import tqdm
import torch, torch.nn as nn, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import ASTForAudioClassification
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast

from src.traintest import set_random_seed

class ShortASTC(ASTForAudioClassification):
    def __init__(self, num_classes: int = 527):
        pretrained_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        super().__init__(pretrained_model.config)

        self.load_state_dict(pretrained_model.state_dict())
        
        if num_classes != 527:
            self.config.num_classes = num_classes
            self.classifier.dense = nn.Linear(self.config.hidden_size, num_classes)

        self.layer_count = len(self.audio_spectrogram_transformer.encoder.layer)

# 测试
# x = torch.rand(4, 1024, 128)
# model = ShortASTC(num_classes=10)
# model(x)['logits'].shape

# region traintest
def train_epoch_amp(device, train_dataloader, model, optimizer, criterion, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=True):
            outputs = model(inputs)['logits']
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(device, val_dataloader, model, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)['logits']
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)  # Adjust for batch-averaged loss
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def fit(
    device,
    trainset: Dataset, 
    valset: Dataset, 
    model: nn.Module = None,        
    batch_size: int = 16,
    num_workers: int = 2,
    lr: float = 1e-5,
    epochs: int = 3
    ):

    trainloader, valloader = get_dataloaders(trainset, valset, batch_size=batch_size, num_workers=num_workers)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)

    bepoch, bacc, bstate = 0, 0, None
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
        val_loss, val_acc = validate(device, valloader, model, criterion)

        if val_acc > bacc:
            bepoch, bacc, bstate = epoch, val_acc, model.state_dict()
        print(f"--Epoch {epoch + 1}/{epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Best Acc {bacc}")
        scheduler.step(val_loss)
    model.load_state_dict(bstate)

    return {
        'state': bstate,
        'param': sum(p.numel() for p in model.parameters()),
        'acc': bacc
    }
# endregion

# region analyse
def analyse_layers(valset: Dataset, model: ShortASTC, key: str='fbank', device: str = "cpu"):
    def angular_distance(v1, v2):
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity = cos_sim(v1, v2)
        similarity = torch.clamp(similarity, -1.0, 1.0)
        return (1 / np.pi) * torch.acos(similarity).mean().item()

    model.eval()
    model.to(torch.device(device))

    layer_count = len(model.audio_spectrogram_transformer.encoder.layer)
    result = np.zeros((layer_count, layer_count))
    with torch.no_grad():
        sample_count = 0
        for d in tqdm(valset, desc="Analyzing layers"):
            outputs = model(
                input_values=d[key].unsqueeze(0).to(device),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            for i in range(layer_count):
                for j in range(i + 1, layer_count):
                    dist = angular_distance(
                        hidden_states[i][:, 0, :],  # [CLS] token
                        hidden_states[j][:, 0, :]
                    )
                    result[j - i - 1, i] += dist
            sample_count += 1

    result /= sample_count
    return result

def get_best_pruning_start(result, block_size: int = 1):
    layer_count = result.shape[0]
    assert block_size < layer_count and block_size > 0
    layer_result = result[block_size, : layer_count - block_size]
    start_layer = np.argmin(layer_result)
    return start_layer

def get_pruning_order(result):
    """
    根据贡献矩阵生成剪枝顺序。

    Args:
        result (np.ndarray): 贡献矩阵，形状为 (block_size, layer_count)。

    Returns:
        list: 剪枝顺序列表，每个元素为 (start_layer, block_size, contribution)。
    """
    order = []
    block_size, layer_count = result.shape

    # 遍历所有可能的剪枝起点和层块大小
    for b in range(1, block_size):
        for l in range(layer_count - b):
            contribution = result[b, l]
            order.append((l, b, contribution))

    # 按贡献值从小到大排序
    order = sorted(order, key=lambda x: x[2])
    return order

def plot_result(result, normalized=True, title="Layer Contribution Heatmap", save_path=None):
    """
    绘制层间贡献热力图，调整序号从 1 开始。

    Args:
        result (np.ndarray): 贡献度矩阵。
        normalized (bool): 是否进行归一化处理。
        title (str): 图像标题。
        save_path (str): 如果指定，将图像保存到文件。
    """
    plt.clf()

    # 构造遮罩
    mask = np.zeros_like(result)
    mask[np.triu_indices_from(mask, k=1)] = True
    mask = np.flip(mask, axis=0)

    # 归一化处理
    if normalized:
        masked_results = np.ma.masked_array(result, mask)
        max_dist = np.ma.max(masked_results, axis=1)[:, np.newaxis]
        min_dist = np.ma.min(masked_results, axis=1)[:, np.newaxis]
        valid_range = max_dist - min_dist
        valid_range[valid_range == 0] = 1  # 避免分母为 0
        result = (result - min_dist) / valid_range

    # 绘制热力图
    ax = sns.heatmap(result, linewidth=0.5, mask=mask, cmap="viridis_r", cbar=True)

    # 设置标签
    x_labels = [str(i + 1) for i in range(result.shape[1])]  # X轴从1开始
    y_labels = [str(i + 1) for i in range(result.shape[0])]  # Y轴从1开始
    ax.set_xticklabels(x_labels, ha="center", fontsize=10)
    ax.set_yticklabels(y_labels, va="center", fontsize=10)

    ax.set_xlabel("Start Layer Index (Pruning Start Point)", fontsize=12)
    ax.set_ylabel("Block Size (Number of Layers Pruned)", fontsize=12)

    # 翻转 Y 轴
    ax.invert_yaxis()

    # 添加标题
    ax.set_title(title, fontsize=14)

    # 显示并保存图像
    plt.tight_layout()  # 调整布局避免标签重叠
    plt.show()
    if save_path:
        plt.savefig(save_path)

def prune_layers(model: ShortASTC, start_layer: int, block_size: int = 1):
    """剪枝指定范围的层"""
    new_layers = torch.nn.ModuleList()
    remove_layers = list(range(start_layer, start_layer + block_size))
    print(f"--Removing layers: {remove_layers}")

    for i in range(0, model.layer_count):
        if i not in remove_layers:
            new_layers.append(model.audio_spectrogram_transformer.encoder.layer[i])

    model.audio_spectrogram_transformer.encoder.layer = new_layers
    model.layer_count -= block_size
    model.config.num_hidden_layers = model.layer_count
    print(f"Pruned model to {model.layer_count} layers.")
    return model

# endregion

if __name__ == '__main__':
    set_random_seed(42)
    from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders

    dataset, num_classes, folds = get_hf_dataset('esc50')
    def one_fold(cuda_id, dataset, fold, epochs):
        trainset, valset = split_hf_dataset(dataset, fold=fold)
        trainloader, valloader = get_dataloaders(trainset, valset, batch_size=16, num_workers=0)
        base_model = ShortASTC(num_classes=num_classes)

        device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
        result = fit(device, trainset, valset, base_model, batch_size=4, lr=1e-5, epochs=epochs)
        base_state, base_param, base_acc = result['state'], result['param'], result['acc']

        params, accs = [base_param], [base_acc]
        base_model.load_state_dict(base_state)
        contribution_array = analyse_layers(valset, base_model, key='fbank', device=device)
        # plot_result(contribution_array, save_path='contribution.png')
        order_list = get_pruning_order(contribution_array)
        for start_layer, block_size, _ in tqdm(order_list, desc="Ordering"):
            model = prune_layers(copy.deepcopy(base_model), start_layer, block_size)
            result = fit(device, trainset, valset, model, batch_size=4, lr=1e-5, epochs=epochs)

            # if result['acc'] < base_acc - 0.02:
            #     break

            params.append(result['param'])
            accs.append(result['acc'])
        return {'params': params, 'accs': accs}

    cuda_id = 1
    all_results = []
    for i in range(1, folds + 1):
        print(f"--Fold {i}:")
        # 获取当前 fold 的结果
        result = one_fold(cuda_id, dataset, i, 30)

        # 展平每个 fold 的结果
        for param, acc in zip(result['params'], result['accs']):
            all_results.append({'fold': i, 'param': param, 'acc': acc})

            torch.cuda.empty_cache()
            gc.collect()

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(all_results)
    df.to_csv('ST.csv', index=False)

    print("Results saved to ST.csv")

    # python run-st.py | tee st.txt