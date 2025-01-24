# 参考torch_pruning.examples.prune_hf_vit.py
import os, sys, argparse, random, copy, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import torch, torch.nn as nn, torch_pruning as tp, numpy as np
from transformers import ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention, ASTSelfOutput, ASTSdpaSelfAttention, ASTIntermediate, ASTOutput

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, fit

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

def main(args):
    # region--base
    set_random_seed(args.seed)

    dataset, num_classes, _ = get_hf_dataset('esc50')
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, num_workers=2, batch_size=32)

    model = get_model(model_name='ast', num_classes=num_classes)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f"cuda:{args.cuda_id}")
    example_inputs = torch.rand(1, 1024, 128).to(device)
    model.to(device)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Base Params:{base_params/1e6:.2f}M, MACs:{base_macs/1e9:.2f}G")
    # endregion

    # region--tune and test before pruning
    if args.test_before_pruning:
        if os.path.exists(args.base_pth):
            model.load_state_dict(torch.load(args.base_pth, weights_only=True))
        else:
            result = fit(device, trainloader, valloader, model, lr=args.lr, epochs=args.epochs)
            model.load_state_dict(result['state'])
        _, base_acc = validate(device, valloader, model, criterion=nn.CrossEntropyLoss())
        print(f"--original acc: {base_acc*100:.2f}%")
    else:
        base_acc = 0
    print(f"Base acc: {100*base_acc:.2f}%")
    # endregion
    #----------------------------------------
    # Pruning
    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    elif args.pruning_type == 'l2':
        imp = tp.importance.MagnitudeImportance(p=2)
    elif args.pruning_type == 'taylor':
        imp = tp.importance.TaylorImportance()
    elif args.pruning_type == 'hessian':
        imp = tp.importance.GroupHessianImportance()        
    else: 
        raise NotImplementedError 

    ignored_layers = [
        model.augmentations,
        model.model.audio_spectrogram_transformer.embeddings,
        model.model.classifier
        ]
    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]

    num_heads = {}
    for m in model.modules():
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

        elif args.bottleneck and isinstance(m, ASTSelfOutput):
            ignored_layers.append(m.dense)

    pruner = tp.pruner.MetaPruner(
                    model,
                    example_inputs,
                    importance=imp, # importance criterion for parameter selection
                    ignored_layers=ignored_layers,
                    unwrapped_parameters=unwrapped_parameters,
                    global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                    iterative_steps=args.iterative_steps,
                    pruning_ratio=args.pruning_ratio, # target pruning ratio
                    # output_transform=lambda out: out.logits.sum(),
                    num_heads=num_heads,
                    prune_head_dims=args.prune_head_dims, # If True, the pruner will prune the same percentage of head dimensions in each iteration.
                    prune_num_heads=args.prune_num_heads, # If True, the pruner will prune the same percentage of heads in each iteration.
                    head_pruning_ratio=args.head_pruning_ratio # disabled when prune_num_heads=False
                    )

    if isinstance(imp, tp.importance.TaylorImportance):
        model.zero_grad()
        print("Accumulating gradients for taylor pruning...")
        for k, (imgs, lbls) in enumerate(trainloader):
            if k>=args.taylor_batchs: break
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            output = model(imgs).logits
            loss = torch.nn.functional.cross_entropy(output, lbls)
            loss.backward()

    for g in pruner.step(interactive=True):
        g.prune()

    # 开始剪枝
    pruner.step()

    # Modify the attention head size and all head size aftering pruning
    for m in model.modules():
        if isinstance(m, ASTSelfAttention):
            # 更新 Self-Attention 的 Head 参数
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features

        elif isinstance(m, torch.nn.Linear):
            # 更新 Linear 层的输入和输出特征
            m.in_features = m.weight.size(1)
            m.out_features = m.weight.size(0)

    pruned_macs, pruned_param = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Pruned params:{pruned_param/1e6:.2f}M, MACs:{pruned_macs/1e9:.2f}G")
    
    #----------------------------------------
    # region Recovery accuracy
    if args.post_train:
        result = fit(device, trainloader, valloader, model, lr=args.lr, epochs=args.epochs)    
        model.load_state_dict(result['state'])
        pruned_acc = result['acc']
    else:
        _, pruned_acc = validate(device, valloader, model, criterion=nn.CrossEntropyLoss())
    print(f"--Pruned acc: {pruned_acc*100:.2f}%")
    # endregion

    return {
        'macs': pruned_macs,
        'param': pruned_param,
        'acc': pruned_acc
    }

def test():
    class Args:
        seed = 42
        fold = 1
        cuda_id = 0
        test_before_pruning = False
        pruning_type = 'l2'
        global_pruning = False
        iterative_steps = 10
        pruning_ratio= 0.5
        prune_head_dims= True
        prune_num_heads= False
        head_pruning_ratio= 0.5
        bottleneck = False
        post_train = False
    args = Args()

    main(args)

def run_trials(args: dict):
    pass

if __name__ == '__main__':
    # region cmds
    import argparse
    parser = argparse.ArgumentParser(description='AST Pruning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--test_before_pruning', action='store_true')
    parser.add_argument('--base_pth', type=str, default='pths/esc50_1-ast-seed_42.pth')
    parser.add_argument('--pruning_type', type=str, default='l2')
    parser.add_argument('--taylor_batchs', type=int, default=10)
    parser.add_argument('--global_pruning', action='store_true')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument('--iterative_steps', type=int, default=10)
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--prune_head_dims', action='store_true')
    parser.add_argument('--prune_num_heads', action='store_true')
    parser.add_argument('--head_pruning_ratio', type=float, default=0.2)
    parser.add_argument('--post_train', action='store_true')
    args = parser.parse_args()
    # endregion
    # main(args)
    test()
