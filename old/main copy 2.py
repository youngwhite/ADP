
# region 
import os, time, copy, tqdm, tempfile, pandas as pd
import torch
from torch.amp import GradScaler
from torch.optim import optimizer

import ray
from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig

from datas.esc_dataloaders import get_hf_dataset, split_hf_dataset, get_dataloaders
from models.model_getter import get_model
from src.traintest import train_epoch_amp, validate_epoch_amp
# endregion

class ADP:
    def __init__(self, dataset_name: str, model_name: str):
        self.task_name = f"ADP-{dataset_name}-{model_name}"

        self.dataset, num_classes, self.folds = get_hf_dataset(dataset_name)
        self.model = get_model(model_name, num_classes)

    @staticmethod
    def train_ray(config: dict, common_space: dict):
        # Data
        dataset = common_space['dataset']
        trainset, valset = split_hf_dataset(dataset, fold=config['fold'])

        trainloader, valloader = get_dataloaders(
            trainset=trainset, valset=valset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        # Model
        model = common_space['model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Optimizer, Criterion, Scheduler
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)
        scaler = GradScaler()

        # Resume from checkpoint if available
        start = 1
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                start = checkpoint_dict["epoch"] + 1
                model.load_state_dict(checkpoint_dict["model_state"])
                print("--Resumed training from checkpoint--")
        # Training
        bepoch, bacc = 0, 0
        for epoch in range(start, config['max_epochs']+1):
            # Training and validation
            tloss, tacc = train_epoch_amp(device, trainloader, model, optimizer, criterion, scaler)
            vloss, vacc = validate_epoch_amp(device, valloader, model, criterion)

            if vacc > bacc:
                bepoch, bacc = epoch, vacc
            metrics = {'tacc': tacc, 'vacc': vacc, 'bacc': bacc, 'bepoch': bepoch, 'tloss': tloss, 'vloss': vloss}

            if vacc > config.get('save_threshold', 0):
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"epoch": epoch, "model_state": model.state_dict()},
                        os.path.join(tempdir, "checkpoint.pt"),
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                train.report(metrics)

            # Log training progress
            print(f"Epoch {epoch}/{config['max_epochs']}, Loss-t/v:{tloss:.4f}/{vloss:.4f}, Acc-t/v:{tacc:.4f}/{vacc:.4f}, "
                  f"Best Acc: {bacc:.4f} at Epoch {bepoch}")

            # Update learning rate scheduler
            scheduler.step(vloss)
            # if epoch >= 5:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.85

        print(f"--Finished Training: Best Epoch {bepoch}, Best Val Acc {bacc:.4f}--")

    @staticmethod
    def tune_ray(trainable: callable, common_space: dict, search_space: dict, resources: dict={'cpu': 2, 'gpu': 1}, num_samples: int=4):
        # ray.init(local_mode=True)
        # os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"
        ray.init()
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(trainable, common_space=common_space),
                resources=config.get('resource_per_trial', {'cpu': 2, 'gpu': 1})
            ),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric='vacc',
                mode='max',
                num_samples=search_space.get('num_samples', 4),
                ),
            run_config=train.RunConfig(
                storage_path=os.path.abspath('ray_results'),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="vacc",
                    checkpoint_score_order="max"
                    )
            )
        )
        result_grid = tuner.fit()
        if not result_grid:
            raise RuntimeError("No results found in tuning process")

        best_result = result_grid.get_best_result(metric="vacc", mode="max", scope='all')

        checkpoint_pth = None
        if best_result.checkpoint:
            best_checkpoint = best_result.get_best_checkpoint('vacc', mode='max')
            with best_checkpoint.as_directory() as checkpoint_dir:
                checkpoint_pth = os.path.join(checkpoint_dir, 'checkpoint.pt')

        ray.shutdown()
        return {
            'config': best_result.config,
            'metrics_df': best_result.metrics_dataframe,
            'hacc': best_result.metrics_dataframe['vacc'].max(),
            'hcheckpoint_pth': checkpoint_pth
        }

    def test_tune(self):
        common_space = {
            'dataset': self.dataset,
            'model': self.model.retain_layers(2),
            'resource_per_trial':{'cpu': 4, 'gpu': 1},
            'num_samples': 4
        }
        search_space = {
            'fold': 1,
            'num_workers': 2,
            'max_epochs': 10,
            'save_threshold': 0,            
            'batch_size': 12,
            'lr': 1e-5
        }
        results = self.tune_ray(
            self.train_ray, 
            common_space=common_space,
            search_space=search_space,
            resources=common_space['resource_per_trial'], 
            num_samples=common_space['num_samples']
            )
        print(f"--Best Acc: {results['hacc']:.4f}, checkpoint_pth: {results['hcheckpoint_pth']}--")

        # test
        config = results['config']
        dataset = common_space['dataset']

        trainset, valset = split_hf_dataset(dataset, fold=search_space['fold'])
        _, valloader = get_dataloaders(
            trainset=trainset, valset=valset,
            num_workers=config['num_workers'], batch_size=config['batch_size']
        )
        
        model = common_space['model']
        sd = torch.load(results['hcheckpoint_pth'], weights_only=True)['model_state']
        if sd:
            model.load_state_dict(sd)
        else:
            raise ValueError("--No checkpoint path found--")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        vloss, vacc = validate_epoch_amp(device, valloader, model, criterion)
        print(f"--Test Accuracy: {vacc:.4f}--")

        # save compressed model
        torch.save(model, f"ADP-vacc_{vacc}.pth")

    def run(self, config):
        common_space = {
            'dataset': self.dataset,
            'model': self.model,
        }

        # train
        all_results = []
        for fold in range(1, self.folds+1):
            # full model
            config['fold'] = fold

            results = self.tune_ray(self.train_ray, common_space=common_space, search_space=config, num_samples=30)
            hacc, hcheckpoint_pth = results['hacc'], results['hcheckpoint_pth']

            all_results.append({
                'fold': fold, 
                'layers': self.model.num_layers, 
                'para': sum(p.numel() for p in self.model.parameters()), 
                'acc': hacc, 
                'checkpoint_pth': hcheckpoint_pth
                })

            # depth reduction training
            for n in range(self.model.num_layers-1, 0, -1):
                self.model.retain_layers(n)

                results = self.tune_ray(self.train_ray, common_space, config)
                acc, checkpoint_pth = results['hacc'], results['hcheckpoint_pth']

                #
                if acc - hacc > config.get('degredation', 0):
                    break
                else:
                    all_results.append({
                        'fold': fold, 
                        'layers': n, 
                        'para': sum(p.numel() for p in self.model.parameters()), 
                        'acc': acc, 
                        'checkpoint_pth': checkpoint_pth
                        })
        pd.DataFrame(all_results).to_csv(f"{self.task_name}.csv", index=False)


if __name__ == '__main__':
    # ADP('esc50', 'ast').test_tune()

    config = {
        'dataset_name': 'esc50',
        'model_name': 'ast',
        'batch_size': 48,        
        'num_workers': 2,
        'lr': 1e-5,
        'max_epochs': 1, #
        'save_threshold': 0.94,
        'degredation': 0.,
        'resources': {'cpu': 4, 'gpu': 1},
        'num_samples': 3 #
    }
    ADP('esc50', 'ast').run(config)

