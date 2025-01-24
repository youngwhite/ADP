# 参考torch_pruning.examples.prune_hf_vit.py
import os, sys, argparse, time, gc, itertools, matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from types import SimpleNamespace

import torch, torch.nn as nn, torch_pruning as tp, numpy as np, pandas as pd
from torch.amp import autocast
from transformers import ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention, ASTSelfOutput, ASTSdpaSelfAttention, ASTIntermediate, ASTOutput

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import set_random_seed, fit

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
    trainset,
    valset,
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

    start = time.time()
    bepoch, bacc, bstate = 0, 0, None
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
        val_loss, val_acc = validate(device, valloader, model, criterion)

        if val_acc > bacc:
            bepoch, bacc, bstate = epoch, val_acc, model.state_dict()
        print(f"--Epoch {epoch + 1}/{epochs}: Loss-t/v: {train_loss:.4f}/{val_loss:.4f}, Acc-t/v: {train_acc:.4f}/{val_acc:.4f}, Best Acc {bacc:.4f} Time: {time.time()-start:.2f}s")
        scheduler.step(val_loss)
    model.load_state_dict(bstate)

    del optimizer
    return {
        'state': bstate,
        'param': sum(p.numel() for p in model.parameters()),
        'acc': bacc
    }
# endregion

# args = SimpleNamespace(cuda_id=0, dataset_name='esc50', model_name='ast', batch_size=32, fold=1, seed=42, lr=1e-5, epochs=3)

def main(args: object):
    # region --basemodel
    device = torch.device(f"cuda:{args.cuda_id}")

    dataset, num_classes, folds = get_hf_dataset(args.dataset_name)
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, num_workers=2, batch_size=args.batch_size)

    example_inputs = torch.rand(1, 1024, 128).to(device)
    # example_inputs, _ = next(iter(valloader))
    # example_inputs = example_inputs.to(device)
    # print(f"--example_inputs.shape: {example_inputs.shape}")
    model = get_model(model_name=args.model_name, num_classes=num_classes)
    model.to(device)
    print(f"--output.shape: {model(example_inputs).logits.shape}")

    base_macs, base_param = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"--Base Params:{base_param/1e6:.2f}M, MACs:{base_macs/1e9:.2f}G")

    if args.test_before_prune:
        pretrained_pth = f"pths/{args.dataset_name}_{args.fold}-{args.model_name}-seed_{args.seed}.pth"
        if os.path.exists(pretrained_pth):
            print(f"--Loading: {pretrained_pth}")
            model.load_state_dict(torch.load(pretrained_pth, map_location=device, weights_only=True))
            _, base_acc = validate(device, valloader, model, nn.CrossEntropyLoss())
        else:
            print(f"--Training Base Model...")
            result = fit(args.cuda_id, trainset, valset, model, lr=args.lr, epochs=args.epochs)
            model.load_state_dict(result['state'])
            _, base_acc = validate(device, valloader, model, criterion)
    else:
        base_acc = 0
    print(f"--Base Acc:{base_acc:.4f}")
    # endregion

    # region --importance
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
    # endregion    

    # region --pruner
    unwrapped_parameters = [
        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
    ]

    num_heads = {}
    ignored_layers = [
        # model.model.audio_spectrogram_transformer.embeddings,
        model.model.classifier
        ]

    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, ASTSdpaSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if args.bottleneck and isinstance(m, ASTSelfOutput):
            ignored_layers.append(m.dense)

    pruner = tp.pruner.MetaPruner(
        model, 
        example_inputs, 
        ignored_layers=ignored_layers,        
        unwrapped_parameters=unwrapped_parameters,
        num_heads=num_heads, # number of heads in self attention        
        importance=imp,
        global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
        iterative_steps=args.iterative_steps,
        pruning_ratio=args.pruning_ratio, # target pruning ratio
        prune_num_heads=args.prune_num_heads,
        prune_head_dims=not args.prune_num_heads, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=1
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

    # Modify the attention head size and all head size aftering pruning
    for m in model.modules():
        if isinstance(m, ASTSdpaSelfAttention):
            print(m)
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()
    
    pruned_macs, pruned_param = tp.utils.count_ops_and_params(model, example_inputs)    
    # print(model)
    # print(f"--output.shape: {model(example_inputs).logits.shape}")
    # endregion

    # region --post_train
    if args.post_train:
        result = fit(args.cuda_id, trainset, valset, model, lr=args.lr, epochs=args.epochs)
        model.load_state_dict(result['state'])
        _, pruned_acc = validate(device, valloader, model, nn.CrossEntropyLoss())
        print(f"--Post Train Acc:{pruned_acc:.4f}")
    else:
        # _, pruned_acc = validate(device, valloader, model, nn.CrossEntropyLoss())
        pruned_acc = 0

    torch.cuda.empty_cache()
    gc.collect()
    del model
    print(">-------------------Summary---------------------")
    print(f"--Base Param: {base_param/1e6:.2f}M, Macs: {base_macs/1e9:.2f}G, Acc: {100*base_acc:.4f}%")
    print(f"--Pruned Param: {pruned_param/1e6:.2f}M, Macs: {pruned_macs/1e9:.2f}G, Acc: {100*pruned_acc:.4f}%")
    print("<------------------------------------------------")
    return {
        'base_param': base_param, 'base_acc': base_acc,
        'pruned_param': pruned_param, 'pruned_acc': pruned_acc
    }

def test(cuda_id: int, epochs: int = 3):
    args = SimpleNamespace(
        dataset_name='esc50', 
        batch_size=16,
        model_name='ast', 
        cuda_id=cuda_id,         
        # fold=1, 
        seed=42, 
        lr=1e-5, 
        epochs=epochs,
        test_before_prune=True,
        post_train=True,
        iterative_steps=1,
        global_pruning=False,
        isomorphic=False,
        # pruning_type='l2',
        # iterative_steps=10,
        # pruning_ratio=0.9,
        prune_num_heads=True,
        prune_head_dims=False,
        head_pruning_ratio=0.5,
        taylor_batchs=10,
        bottleneck=True
        )

    set_random_seed(args.seed)
    # IP
    all_results = []
    for fold in [1, 2]:
        args.fold = fold
        for pruning_type in ['taylor', 'l2']:
            args.pruning_type = pruning_type
            for pruning_ratio in [0.1, 0.2]:
                args.pruning_ratio = pruning_ratio
                result = main(args)
                result['fold'] = fold
                result['pruning_type'] = pruning_type
                result['pruning_ratio'] = pruning_ratio
                all_results.append(result)

                torch.cuda.empty_cache()
                gc.collect()
    pd.DataFrame(all_results).to_csv('results-test.csv', index=False)

def run_experiment_local(cuda_id: int, epochs: int = 3):
    args = SimpleNamespace(
        dataset_name='esc50', 
        batch_size=16,         
        model_name='ast', 
        cuda_id=cuda_id,         
        # fold=1, 
        seed=42, 
        lr=1e-5, 
        epochs=epochs,
        test_before_prune=True,
        post_train=True,
        iterative_steps=1,
        global_pruning=False,
        isomorphic=False,
        # pruning_type='l2',
        # iterative_steps=10,
        # pruning_ratio=0.9,
        prune_num_heads=False,
        prune_head_dims=True,
        head_pruning_ratio=0.5,
        taylor_batchs=10,
        bottleneck=True
        )

    # 参数组合
    folds = [1, 2, 3, 4, 5]
    pruning_types = ['l2', 'taylor']
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_combinations = itertools.product(folds, pruning_types, pruning_ratios)

    # 收集结果
    all_results = []
    for fold, pruning_type, pruning_ratio in tqdm(param_combinations, desc='Running Local Pruning'):
        args.fold = fold
        args.pruning_type = pruning_type
        args.pruning_ratio = pruning_ratio

        # 运行实验
        result = main(args)
        result['fold'] = fold
        result['pruning_type'] = pruning_type
        result['pruning_ratio'] = pruning_ratio
        all_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(all_results).to_csv('pruner-local.csv', index=False)

def run_experiment_global(cuda_id: int, epochs: int = 3):
    args = SimpleNamespace(
        dataset_name='esc50', 
        batch_size=16,         
        model_name='ast', 
        cuda_id=cuda_id,         
        # fold=1,
        seed=42, 
        lr=1e-5, 
        epochs=epochs,
        test_before_prune=True,
        post_train=True,
        iterative_steps=1,
        global_pruning=True, ###
        isomorphic=False,
        # iterative_steps=10,
        # pruning_ratio=0.9,
        prune_num_heads=False,
        prune_head_dims=True,
        head_pruning_ratio=0.5,
        taylor_batchs=10,
        bottleneck=True,
        # pruning_type='random'
        )

    # IP
    # 参数组合
    folds = [1, 2, 3, 4, 5]
    pruning_types = ['l2', 'taylor']
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_combinations = itertools.product(folds, pruning_types, pruning_ratios)

    # 收集结果
    all_results = []
    for fold, pruning_type, pruning_ratio in tqdm(param_combinations, desc='Running Local Pruning'):
        args.fold = fold
        args.pruning_type = pruning_type
        args.pruning_ratio = pruning_ratio

        # 运行实验
        result = main(args)
        result['fold'] = fold
        result['pruning_type'] = pruning_type
        result['pruning_ratio'] = pruning_ratio
        all_results.append(result)
        
        torch.cuda.empty_cache()
        gc.collect()        

    pd.DataFrame(all_results).to_csv('results-global.csv', index=False)

def run_experiment_isomorphic(cuda_id: int, epochs: int = 3):
    args = SimpleNamespace(
        dataset_name='esc50', 
        batch_size=16,         
        model_name='ast', 
        cuda_id=cuda_id,
        # fold=1,
        seed=42, 
        lr=1e-5, 
        epochs=epochs,
        test_before_prune=True,
        post_train=True,
        iterative_steps=1,        
        global_pruning=True,
        isomorphic=True, ###
        # iterative_steps=10,
        # pruning_ratio=0.9,
        prune_num_heads=False,
        prune_head_dims=True,
        head_pruning_ratio=0.5,
        taylor_batchs=10,
        bottleneck=True,
        # pruning_type='random'
        )

    # IP
    all_results = []
    for fold in [1, 2, 3, 4, 5]:
        args.fold = fold
        for pruning_type in ['l2', 'taylor']:
            args.pruning_type = pruning_type
            for pruning_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                args.pruning_ratio = pruning_ratio
                result = main(args)
                result['fold'] = fold
                result['pruning_type'] = pruning_type
                result['pruning_ratio'] = pruning_ratio
                all_results.append(result)

                torch.cuda.empty_cache()
                gc.collect()
    pd.DataFrame(all_results).to_csv('results-iso.csv', index=False)

if __name__ == '__main__':
    # region args
    parser = argparse.ArgumentParser(description='hf Pruner')
    parser.add_argument('--dataset_name', type=str, default='esc50', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_name', type=str, default='ast', help='model name')
    
    parser.add_argument('--cuda_id', type=int, default=0, help='cuda id')
    parser.add_argument('--fold', type=int, default=2, help='fold')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')

    parser.add_argument('--test_before_prune', action='store_true')
    parser.add_argument('--post_train', action='store_true')
    parser.add_argument('--global_pruning', action='store_true')
    parser.add_argument('--iterative_steps', type=int, default=1, help='iterative steps')
    parser.add_argument('--pruning_ratio', type=float, default=0.9, help='pruning ratio')
    parser.add_argument('--prune_num_heads', action='store_true')
    parser.add_argument('--prune_head_dims', action='store_true')
    parser.add_argument('--head_pruning_ratio', type=float, default=0.9, help='head pruning ratio')
    parser.add_argument('--taylor_batchs', type=int, default=10, help='taylor batchs')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument('--pruning_type', type=str, default='random', help='pruning type')
    args = parser.parse_args()
    # endregion

    main(args)
    # python run-hf-pruner.py --fold 2 --test_before_prune --pruning_ratio 0.9 --head_pruning_ratio 0.5
    set_random_seed(42)
    # test(args.cuda_id)
    # run_experiment_local(cuda_id=0, epochs=30)
    run_experiment_global(cuda_id=1, epochs=30)
    # python run-hf-pruner.py | tee global.txt

    # run_experiment_isomorphic(cuda_id=2, epochs=30)
    # python run-hf-pruner.py | tee iso.txt
