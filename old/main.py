import time, copy
import torch
from torch.amp import GradScaler
from torch.optim import optimizer

from datas.esc_dataloaders import get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate_epoch_amp

def train(config: dict):
    prefix = config['prefix']

    train_loader, val_loader = config['trainloader'], config['valoader']
    train_loader, val_loader = ray.train.torch.prepare_data_loader(train_loader), ray.train.torch.prepare_data_loader(val_loader)

    model = config['model']
    model = ray.train.torch.prepare_model(model)

    t0, patience, progress = time.time(), 0, []
    bepoch, bacc, bmodel_state = 0, 0, None
    for epoch in range(config['max_epochs']):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)        

        tloss, tacc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
        vloss, vacc = validate_epoch_amp(device, valoader, model, criterion)

        if vacc > bacc:
            bepoch, bmodel_state, bacc = epoch, copy.deepcopy(model.state_dict()), vacc

            patience = 0
        else:
            patience += 1

        # print(f"--{prefix}, Epoch:{epoch+1}/{config['max_epochs']}, loss:{tloss:.4f}/{vloss:.4f}, "
        #         f"tacc:{tacc:.4f}, vacc:{vacc:.4f}, "
        #         f"bacc:{bacc:.4f}, bepoch:{bepoch}, patience:{patience}, "
        #         f"lr:{optimizer.param_groups[0]['lr']:.1e}, time:{time.time()-t0:.0f}s")
        
        metrics = {'epoch': epoch, 'train_loss': tloss, 'train_acc': tacc, 'val_loss': vloss, 'val_acc': vacc, 'patience': patience}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)        
        
        # progress.append(metrics)
        # scheduler.step(vloss)

        # if patience > 30:
        #     break
    # return {'config': config, 'progress': progress, 'bacc': bacc, 'bmodel_state': bmodel_state}

def trials(
    trainable: callable, 
    config_tune: dict,
    resource_per_trial: dict, 
    num_samples: int, 
    storage_path: str='ray_results'
    ) -> dict:
    if not ray.is_initialized():
        # os.system("pkill -f 'ray'")
        ray.init(address=None, ignore_reinit_error=True)
    run_config = train.RunConfig(
        # name=,
        storage_path=os.path.abspath(storage_path),            
        log_to_file=False,
        # stop=TrialPlateauStopper(
        #     metric='vacc',
        #     std=0.01,
        #     num_results=10,
        #     metric_threshold=0.90,
        #     mode='max'
        #     ),            
        failure_config=FailureConfig(max_failures=-1, fail_fast=False),
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="vacc",
            checkpoint_score_order="max"),
        callbacks=[self.trial_callback]
        )
    tune_config = tune.TuneConfig(
        metric="vacc",
        mode="max",
        num_samples=num_samples,
        # scheduler=asha_scheduler, #不能和tune_config的metric同时使用
        reuse_actors=True # 重用 actor
        )

    tuner = tune.Tuner(
        trainable=tune.with_resources(trainable, resources=resource_per_trial),
        param_space=config_tune,
        tune_config=tune_config,
        run_config=run_config
    )

    result_grid = tuner.fit()

    ray.shutdown()
    return result_grid


def ADP(config: dict):
    trainset, valset, num_classes = get_dataloaders(config)

    config['num_classes'] = num_classes
    model = get_model(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler('cuda')

    #
    for num_layers in range(model.num_layers, 0, -1):
        print(f"Layer:{num_layers}")

if __name__ == '__main__':
    config = {
        'prefix': 'AST-1',
        'dataset': 'esc50',
        'fold': 1,
        'model': 'AST',
        'layer_num': 1,
        'num_classes': 50,
        'lr': 1e-4,
        'batch_size': 128,
        'max_epochs': 10
    }

    # train(config)
    ADP(config)
