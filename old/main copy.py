import os, time, copy, tqdm, tempfile, pandas as pd
import torch
from torch.amp import GradScaler
from torch.optim import optimizer

import ray
from ray import train, tune
from ray.train import Checkpoint

from datas.esc_dataloaders import get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate_epoch_amp


def train_ray(config: dict):
    trainloader, valloader, num_classes = get_dataloaders(config)
    model = get_model(config)
    model.retain_layers(config['layer_num'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler()

    ## Training
    bepoch, bacc = 0, 0
    for epoch in range(config['max_epochs']):
        tloss, tacc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
        vloss, vacc = validate_epoch_amp(device, valloader, model, criterion)

        if vacc > bacc:
            bepoch, bacc, bmodel_state = epoch, vacc, copy.deepcopy(model.state_dict())

        metrics = {'tacc': tacc, 'vacc': vacc, 'tloss': tloss, 'vloss': vloss, 'bepoch': bepoch, 'bacc': bacc}
        print(f"epoch: {epoch}, vacc: {vacc}")
        train.report(metrics)

        scheduler.step(vloss)
    
    torch.save(bmodel_state, f"esc50_{config['fold']}-AST_layer{config['layer_num']}-{bacc:.4f}.pth")
    print('--Finished Training--')

def tune_ray(trainable, param_space, resource_per_trial={'cpu': 2, 'gpu': 1}, num_samples=10):
    if not ray.is_initialized():
        ray.init(address=None, ignore_reinit_error=True)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable),
            resource_per_trial
        ),
        tune_config=tune.TuneConfig(metric='vacc', mode='max', num_samples=num_samples),
        param_space=param_space
    )
    results = tuner.fit()
    best_result = results.get_best_result("vacc", mode="max")

    return best_result 

def ADP_ray(config: dict):
    bresults = []
    model = get_model(config)

    accs = []
    for num_layers in range(model.num_layers, 0, -1):
        acc5 = {}
        for fold in range(1, 6):
            config['fold'] = fold
            config['layer_num'] = num_layers

            best_result = tune_ray(
                train_ray, 
                config, 
                resource_per_trial={'cpu': 2, 'gpu': 1}, num_samples=4
                )
            acc5[i] = best_result.metrics['vacc']
            best_result.metrics_dataframe.to_csv(f"results/{config['dataset']}-{config['model']}-layer_{num_layers}-fold_{fold}.csv")
            print(f'--Layer_{num_layers}, Fold_{fold}: {best_result.metrics["vacc"]}')
        accs.append(acc5)
    pd.DataFrame(accs).to_csv(f"results/{config['dataset']}_ADP_accs.csv")
    print('--Finished ADP--')

if __name__ == '__main__':
    config = {
        # 'dataset': 'ESC-50',
        'dataset': 'UrbanSound8K',
        'fold': 1,
        'num_workers': 2,
        'model': 'AST',
        'num_classes': 50,
        'lr': 1e-5,
        'batch_size': 12,
        'max_epochs': 100
    }
    # train_ray(config)
    ADP_ray(config)

