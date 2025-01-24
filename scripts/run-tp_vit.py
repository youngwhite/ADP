# 参考torch_pruning.examples.prune_hf_vit.py
import os, sys, argparse, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import torch, torch.nn as nn, torch_pruning as tp, numpy as np
from transformers import ASTForAudioClassification
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTSelfAttention, ASTSelfOutput

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate

def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# class Args:
#     fold = 1
#     lr = 1e-5
#     epochs = 3
#     pruning_type = 'l1'
#     pruning_ratio = 0.5
#     global_pruning = True
#     bottleneck = False
#     taylor_batchs = 10
# args = Args()

def main(args: dict):
    set_random_seed()

    dataset, num_classes, folds = get_hf_dataset('esc50')
    trainset, valset = split_hf_dataset(dataset, fold=args.fold)
    trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, num_workers=4, batch_size=16)

    model = get_model(model_name='ast', num_classes=50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    example_inputs = torch.randn(1, 1024, 128).to(device)
    model.to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Base MACs: {base_macs/1e9:.2f}G Params: {base_params/1e6:.2f}M")

    def fit(
        device,
        trainloader,
        valloader,
        model,
        lr=1e-5,
        epochs=100
        ):
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
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
        return {'epoch': bepoch, 'bacc': bacc, 'bstate': bstate}

    pth = f"pths/esc50_{args.fold}-ast-seed_{args.seed}.pth"
    if os.path.exists(pth):
        model.load_state_dict(torch.load(pth, weights_only=True))
        _, base_acc = validate(device, valloader, model, criterion=nn.CrossEntropyLoss())
        print(f"--original acc: {base_acc:.2f}%")
    else:
        result = fit(device, trainloader, valloader, model, lr=args.lr, epochs=args.epochs)
        torch.save(result['bstate'], pth)

        base_acc = result['bacc']
        model.load_state_dict(result['bstate'])
        print(f"--original acc: {base_acc*100:.2f}%")

    #----------------------------------------
    # Pruning
    if args.pruning_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruning_type == 'taylor':
        imp = tp.importance.TaylorImportance()
    elif args.pruning_type == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    else: 
        raise NotImplementedError

    num_heads = {}
    ignored_layers = [
        model.model.audio_spectrogram_transformer.embeddings,
        model.model.classifier
        ]
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, ASTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if args.bottleneck and isinstance(m, ASTSelfOutput):
            ignored_layers.append(m.dense)

    pruner = tp.pruner.MetaPruner(
                    model, 
                    example_inputs, 
                    global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                    importance=imp, # importance criterion for parameter selection
                    pruning_ratio=args.pruning_ratio, # target pruning ratio
                    ignored_layers=ignored_layers,
                    # output_transform=lambda out: out.logits.sum(),
                    num_heads=num_heads,
                    prune_head_dims=True,
                    prune_num_heads=False,
                    head_pruning_ratio=0.5, # disabled when prune_num_heads=False
                    unwrapped_parameters=[
                        (model.model.audio_spectrogram_transformer.embeddings.cls_token, 0),
                        (model.model.audio_spectrogram_transformer.embeddings.distillation_token, 0),
                        (model.model.audio_spectrogram_transformer.embeddings.position_embeddings, 0),
                        ]
                    )

    if isinstance(imp, tp.importance.TaylorImportance):
        model.zero_grad()
        print("Accumulating gradients for taylor pruning...")
        for k, (imgs, lbls) in enumerate(train_loader):
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
        if isinstance(m, ASTSelfAttention):
            print(m)
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size, '=>')
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features
            print("num_heads:", m.num_attention_heads, 'head_dims:', m.attention_head_size, 'all_head_size:', m.all_head_size)
            print()


    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Pruned MACs: {pruned_macs/1e9:.2f}G Params: {pruned_params/1e6:.2f}M")
    #----------------------------------------
    # Recovery accuracy
    if args.post_train:
        result = fit(device, trainloader, valloader, model, lr=args.lr, epochs=args.epochs)    
        pruned_acc = result['bacc']
        model.load_state_dict(result['bstate'])
        print(f"--pruned acc: {pruned_acc*100:.2f}%")
    else:
        _, pruned_acc = validate(device, valloader, model, criterion=nn.CrossEntropyLoss())
        print(f"--pruned acc: {pruned_acc*100:.2f}%")
    #----------------------------------------
    print("Summary:")
    print(f"Base MACs: {base_macs/1e9:.2f}G, Params: {base_params/1e6:.2f}M, Acc: {base_acc*100:.2f}%")
    print(f"Pruned MACs: {pruned_macs/1e9:.2f}G, Params: {pruned_params/1e6:.2f}M, Acc: {pruned_acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AST Pruning')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--fold', type=int, default=1, help='fold index')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--pruning_type', type=str, default='l1', help='pruning type')
    parser.add_argument('--group_reduction', type='first', help='pruning ratio')
    parser.add_argument('--pruning_ratio', type=float, help='pruning ratio')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='bottleneck or uniform')
    parser.add_argument('--global_pruning', action='store_true', help='global pruning')
    parser.add_argument('--taylor_batchs', type=int, default=10, help='taylor batchs')
    parser.add_argument('--post_train', action='store_true', help='post train')
    args = parser.parse_args()

    main(args)
    # test
    # python others/run-tp_vit.py --fold 1 --lr 1e-5 --epochs 3 --pruning_type l1 --pruning_ratio 0.5 --global_pruning --bottleneck --taylor_batchs 10 --post_train