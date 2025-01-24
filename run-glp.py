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
    cuda_id: int,
    trainset: Dataset,
    valset: Dataset,
    model: nn.Module = None,
    batch_size: int = 32,
    num_workers: int = 2,
    lr: float = 1e-5,
    epochs: int = 3
    ):

    trainloader, valloader = get_dataloaders(trainset, valset, batch_size=batch_size, num_workers=num_workers)

    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
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

# region greedy_layer_pruning

def layer_pruning(cuda_id: int, valloader: Dataset, model: ShortASTC):
    # 指定设备
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    
    def remove_single_layer(model: ShortASTC, layer_id: int):
        # 创建新的层列表
        new_layers = torch.nn.ModuleList()
        for i in range(model.layer_count):
            if i != layer_id:
                new_layers.append(model.audio_spectrogram_transformer.encoder.layer[i])

        # 更新模型的层和配置
        model.audio_spectrogram_transformer.encoder.layer = new_layers
        model.layer_count -= 1
        model.config.num_hidden_layers = model.layer_count
        return model

    # 损失函数
    ce_loss = nn.CrossEntropyLoss()

    # 初始化每层的准确率字典
    id2score = {id: 0.0 for id in range(model.layer_count)}

    # 逐层剪枝并评估
    for id in id2score.keys():
        # 使用模型的深拷贝，避免影响原模型
        submodel = remove_single_layer(copy.deepcopy(model), id)

        # 验证当前模型的准确率
        _, id2score[id] = validate(device, valloader, submodel, ce_loss)
        print(f"--Layer {id} Acc: {id2score[id]:.4f}")

    # 找出对性能贡献最小的层（得分最高的层）
    id_to_prune = max(id2score, key=id2score.get)
    print(f"Pruning layer {id_to_prune} with score {id2score[id_to_prune]:.4f}")

    # 剪枝并返回新模型
    pruned_model = remove_single_layer(copy.deepcopy(model), id_to_prune)
    return pruned_model

def greedy_layer_pruning(cuda_id: int, dataset, fold: int, epochs: int, acc_drop: float=0.2):
    trainset, valset = split_hf_dataset(dataset, fold=fold)
    trainloader, valloader = get_dataloaders(trainset, valset, batch_size=32, num_workers=0)

    model = ShortASTC(num_classes=num_classes)

    result = fit(cuda_id, trainset, valset, model, batch_size=32, lr=1e-5, epochs=epochs)
    base_state, base_param, base_acc = result['state'], result['param'], result['acc']
    model.load_state_dict(base_state)
    print(f"--Base Parameters: {base_param}, Acc: {base_acc:.4f}")

    params, accs = [base_param], [base_acc]
    while True:
        submodel = layer_pruning(cuda_id, valloader, copy.deepcopy(model))
        result = fit(cuda_id, trainset, valset, submodel, batch_size=32, lr=1e-5, epochs=epochs)
        pruned_param, pruned_acc = result['param'], result['acc']

        if result['acc'] < base_acc - acc_drop:
            break
        model = submodel
        params.append(pruned_param)
        accs.append(pruned_acc)

    print(f"--Final Pruned Model Params: {pruned_param}, Acc: {pruned_acc:.4f}")
    return {'params': params, 'accs': accs}
# endregion

if __name__ == '__main__':
    from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders

    dataset, num_classes, folds = get_hf_dataset('esc50')

    all_results = []
    for i in range(1, folds+1):
        print(f"--Fold {i}:")

        result = greedy_layer_pruning(cuda_id=1, dataset=dataset, fold=i, epochs=40)

        for param, acc in zip(result['params'], result['accs']):
            all_results.append({'fold': i, 'param': param, 'acc': acc})

    pd.DataFrame(all_results).to_csv('GLP.csv')
    print("Results saved to GLP.csv")
    # python run-glp.py > glp.txt
