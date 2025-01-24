import os, sys, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Optional, Union
from functools import partial, wraps
from tqdm import tqdm
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, ASTForAudioClassification
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast

from src.traintest import set_random_seed, train_epoch_amp, validate_epoch_amp

class ShortASTC(ASTForAudioClassification):
    def __init__(self, num_labels: int = 527):
        pretrained_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        super().__init__(pretrained_model.config)

        self.load_state_dict(pretrained_model.state_dict())
        
        if num_labels != 527:
            self.config.num_labels = num_labels
            self.classifier = self.classifier.__class__(self.config)

        self.layer_count = len(self.audio_spectrogram_transformer.encoder.layer)
        self.param, self.acc, self.bstate = 0, 0, None

    def fit(
        self, 
        trainset: Dataset, 
        valset: Dataset, 
        batch_size: int = 1,
        num_workers: int = 4,
        lr: float = 1e-5,
        epochs: int = 3
        ):

        trainloader, valloader = get_dataloaders(trainset, valset, batch_size=batch_size, num_workers=num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler()        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)

        bepoch, bacc, bstate = 0, 0, None
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch_amp(device, trainloader, self, optimizer, criterion, scaler)
            val_loss, val_acc = validate_epoch_amp(device, valloader, self, criterion)

            if val_acc > bacc:
                bepoch, bacc, bstate = epoch, val_acc, self.state_dict()
            print(f"--Epoch {epoch + 1}/{epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Best Acc {bacc}")
            scheduler.step(val_loss)

        self.bstate = bstate
        self.param, self.acc = sum(p.numel() for p in self.parameters()), bacc
        return self

    @staticmethod
    def angular_distance(v1, v2):
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity = cos_sim(v1, v2)
        similarity = torch.clamp(similarity, -1.0, 1.0)
        return (1 / np.pi) * torch.acos(similarity).mean().item()

    def analyse_layers(self, valset: Dataset, key: str='fbank', device: str = "cpu"):
        self.eval()
        self.to(torch.device(device))

        layer_count = self.layer_count

        result = np.zeros((layer_count, layer_count))
        with torch.no_grad():
            sample_count = 0
            for d in tqdm(valset, desc="Analyzing layers"):
                outputs = self(
                    input_values=d[key].unsqueeze(0).to(device),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states

                for i in range(layer_count):
                    for j in range(i + 1, layer_count):
                        dist = self.angular_distance(
                            hidden_states[i][:, 0, :],  # [CLS] token
                            hidden_states[j][:, 0, :]
                        )
                        result[j - i - 1, i] += dist
                sample_count += 1

        result /= sample_count
        return result

    def plot_result(self, result, normalized=True, title="Layer Contribution Heatmap", save_path=None):
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

    def get_best_pruning_start(self, result, block_size: int = 1):
        layer_count = result.shape[0]
        assert block_size < layer_count and block_size > 0
        layer_result = result[block_size, : layer_count - block_size]
        start_layer = np.argmin(layer_result)
        return start_layer

    def get_pruning_order(self, result):
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

    def prune_layers(self, start_layer: int, block_size: int = 1):
        """
        剪枝指定范围的层。
        """
        new_layers = torch.nn.ModuleList()
        remove_layers = list(range(start_layer, start_layer + block_size))
        print(f"--Removing layers: {remove_layers}")

        for i in range(0, self.layer_count):
            if i not in remove_layers:
                new_layers.append(self.audio_spectrogram_transformer.encoder.layer[i])

        self.audio_spectrogram_transformer.encoder.layer = new_layers
        self.layer_count -= block_size
        self.config.num_hidden_layers = self.layer_count
        print(f"Pruned model to {self.layer_count} layers.")
        return self

if __name__ == '__main__':
    from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders

    # 获取 ESC 数据集和 DataLoader
    dataset, num_classes, folds = get_hf_dataset('esc50')

    for fold in range(1, folds+1):
        print(f"--Fold {fold}:")
        trainset, valset = split_hf_dataset(dataset, fold=fold)
        print(f"Trainset size: {len(trainset)}, Valset size: {len(valset)}")

        set_random_seed(42)
        # 初始化模型
        model = ShortASTC(num_labels=num_classes)
        model.fit(trainset, valset, batch_size=16, lr=1e-5, epochs=100)
        torch.save(model.bstate, f"fold_{fold}_{str(model.param)}_{str(model.acc)}.pth")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 单样本推理
        x = valset[0]["fbank"].unsqueeze(0).to(device)
        y = model(x, output_hidden_states=True)
        print("--Hidden states length:", len(y.hidden_states))
        print("--Logits shape:", y.logits.shape)

        # 分析层贡献
        print("--Analyzing layers...")
        result = model.analyse_layers(valset, device="cuda")

        # 绘制热力图
        print("--Plotting results...")
        model.plot_result(result, save_path = 'contribution.png')

        # 获取剪枝顺序
        order = model.get_pruning_order(result)
        print("--Pruning order:", order)

        # 剪枝并测试推理
        print("--Finding best pruning start layer...")
        start_layer = model.get_best_pruning_start(result)
        print(f"--Best pruning start layer: {start_layer}")

        # 剪枝模型
        model = model.prune_layers(start_layer=start_layer, block_size=1)

        # 测试剪枝后的推理
        model.fit(trainset, valset, batch_size=16, lr=1e-5, epochs=100)
        torch.save(model.bstate, f"fold_{fold}_{str(model.param)}_{str(model.acc)}.pth")
