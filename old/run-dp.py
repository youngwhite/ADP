import torch, torch.nn as nn
import torch_pruning as tp
from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate_epoch_amp, validate

example_inputs = torch.randn(1, 1024, 128)  # 示例输入
model = get_model(model_name='ast', num_classes=50)

base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
print(f"Base MACs: {base_macs/1e9:.0f}G Params: {base_params/1e6:.0f}M")

## 测试
dataset, num_classes, folds = get_hf_dataset('esc50')

trainset, valset = split_hf_dataset(dataset, fold=1)
trainloader, valloader = get_dataloaders(trainset=trainset, valset=valset, num_workers=4, batch_size=3)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
        val_loss, val_acc = validate_epoch_amp(device, valloader, model, criterion)

        if val_acc > bacc:
            bepoch, bacc, bstate = epoch, val_acc, model.state_dict()
        print(f"--Epoch {epoch + 1}/{epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Best Acc {bacc}")
        scheduler.step(val_loss)
    return {'epoch': bepoch, 'acc': bacc, 'state': bstate}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = fit(device, trainloader, valloader, model, epochs=3)
model.load_state_dict(d['state'])
acc = validate(device, valloader, model)
print(f"--original acc: {acc}")

example_inputs = example_inputs.to(device)
DG = tp.DependencyGraph().build_dependency(model, example_inputs)

imp = tp.importance.GroupNormImportance(p=2)  # L2 norm importance

ignored_layers = [
    model.model.audio_spectrogram_transformer.embeddings, 
    model.model.audio_spectrogram_transformer.embeddings.cls_token,
    model.model.audio_spectrogram_transformer.embeddings.distillation_token,
    model.model.audio_spectrogram_transformer.embeddings.position_embeddings,
    model.model.classifier
    ]

ffn_layers, num_heads = [], {}
for layer in model.model.audio_spectrogram_transformer.encoder.layer:
    num_heads[layer.attention.attention.query] = 12  # Assume 12 attention heads
    ffn_layers.append(layer.intermediate.dense)

pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.05,
    global_pruning=True,
    ignored_layers=ignored_layers,
    num_heads=num_heads,
    prune_num_heads=True,
    prune_head_dims=False,
    round_to=8,
)

# Perform pruning
for i, group in enumerate(pruner.step(interactive=True)):
    group.prune()

# Update `num_heads` and related parameters
for idx, layer in enumerate(model.model.audio_spectrogram_transformer.encoder.layer):
    layer.attention.attention.num_heads = pruner.num_heads[layer.attention.attention.query]
    layer.attention.attention.head_dim = (
        layer.attention.attention.query.out_features // layer.attention.attention.num_heads
    )

# Recount MACs and parameters
pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
print(f"Base MACs: {base_macs/1e9} G -> Pruned MACs: {pruned_macs/1e9} G")
print(f"Base Params: {base_params/1e6} M -> Pruned Params: {pruned_params/1e6} M")

# Validate the pruned model
print("Testing the pruned model...")
acc = validate(device, valloader, model)
print(f"--pruned acc: {acc}")
