import os, sys, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, torch.nn as nn, torch_pruning as tp
from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate, set_random_seed, fit

def main(args):
    set_random_seed(args.seed)

    dataset, num_classes, folds = get_hf_dataset('esc50')
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name='ast', num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    pth = f"pths/esc50_{args.fold}-ast-seed_{args.seed}.pth"
    if os.path.exists(pth):
        model.load_state_dict(torch.load(pth, weights_only=True))
        print("--Loaded model from file")
        _, acc = validate(device, valloader, model, criterion)
    else:
        print("--Training model...")
        result = fit(device, trainloader, valloader, model, epochs=args.epochs)
        torch.save(model.state_dict(), pth)
        print("--Model saved to file")
        acc = result['acc']
    print(f"--acc: {100*acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)

    # CUDA_VISIBLE_DEVICES=1 python baseline.py --fold 1 --seed 42