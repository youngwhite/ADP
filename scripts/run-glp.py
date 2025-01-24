import os, sys, argparse, copy, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn as nn, torch_pruning as tp
from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate, set_random_seed, fit

def greedy_layer_pruning(device, trainloader, valloader, model, acc_drop: float=0.2):
    example_inputs = torch.randn(1, 1024, 128).to(device)
    model.to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    ce_loss = nn.CrossEntropyLoss()
    _, base_acc = validate(device, valloader, model, ce_loss)
    print(f"--Base MACs: {base_macs/1e9:.2f}G Params: {base_params/1e6:.2f}M Acc: {base_acc:.4f}")

    ## Greedy Layer Pruning
    while True:
        
        # scoring layers
        id2score = {id: 0. for id in range(model.num_layers)}
        for id in id2score.keys():
            shadow_model = copy.deepcopy(model)
            shadow_model.remove_single_layer(id)
            _, id2score[id] = validate(device, valloader, shadow_model, ce_loss)
            # d = fit(device, trainloader, valloader, shadow_model, lr=1e-5, epochs=60)
            # id2score[id] = d['acc']

        print(f"--Layer Scores: {id2score}")

        # 找到最大准确率对应的层
        id_to_prune = max(id2score, key=id2score.get)

        model.remove_single_layer(id_to_prune)
        print(f"--Pruned Layer {id_to_prune}")
        # _, pruned_acc = validate(device, valloader, model, ce_loss)
        d = fit(device, trainloader, valloader, model, lr=1e-5, epochs=60)
        pruned_acc = d['acc']

        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"--Pruned MACs: {pruned_macs/1e9:.2f}G, Params: {pruned_params/1e6:.2f}M, Acc: {pruned_acc:.4f}")
        if pruned_acc < base_acc - acc_drop:
            break

    print(f"--Final Pruned Model Acc: {pruned_acc:.4f}")

def main(args):
    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset, num_classes, folds = get_hf_dataset('esc50')
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset, valset, batch_size=32)

    model = get_model(model_name='ast', num_classes=50)
    pth = f"pths/esc50_{args.fold}-ast-seed_{args.seed}.pth"
    if os.path.exists(pth):
        model.load_state_dict(torch.load(pth, weights_only=True))
    else:
        raise FileNotFoundError(f"--Weights not found at {pth}")

    greedy_layer_pruning(device, trainloader, valloader, model, acc_drop=args.acc_drop)
    print("--Post Training--")

    print('--Done--')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--acc_drop', type=float, default=0.02)
    args = parser.parse_args()
    main(args)

    # CUDA_VISIBLE_DEVICES=1 python others/run-glp.py --fold 1 --seed 42 --acc_drop 0.2
