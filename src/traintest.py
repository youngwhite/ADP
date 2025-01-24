import time, random, torch, torch.nn as nn, numpy as np
from torch.amp import autocast

def set_random_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch_amp(device, train_dataloader, model, optimizer, criterion, scaler):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=True):
            outputs = model(inputs)
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

def validate_epoch_amp(device, val_dataloader, model, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Use autocast only if AMP is required
            with autocast(device_type='cuda', enabled=True):
                outputs = model(inputs)
                loss = criterion(outputs.float(), labels)  # Ensure type compatibility

            total_loss += loss.item() * len(labels)  # Adjust for batch-averaged loss
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # Optionally log the metrics
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

    return avg_loss, accuracy

def train_epoch(device, train_dataloader, model, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)  # Adjust for batch-averaged loss
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def fit(
    device,
    trainloader, 
    valloader,
    model: nn.Module = None,        
    batch_size: int = 32,
    num_workers: int = 2,
    lr: float = 1e-5,
    epochs: int = 3
    ):
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
