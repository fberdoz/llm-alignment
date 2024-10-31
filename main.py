import os
import time
import torch.optim as optim
import wandb
from helpers import *
from models import *
from torch.utils.data import DataLoader
import argparse
import yaml
from datetime import datetime
import logging

PROJECT_NAME = 'llm-alignment'


def set_logger(name, verbose='info'):
    logger = logging.getLogger(name)

    if verbose == 'debug':
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    elif verbose == 'info':
        logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    elif verbose == 'warning':
        logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stdout)
    elif verbose == 'error':
        logging.basicConfig(level=logging.ERROR, format='%(message)s', stream=sys.stdout)
    elif verbose == 'critical':
        logging.basicConfig(level=logging.CRITICAL, format='%(message)s', stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
        logger.warning(f"Invalid logging level '{verbose}'. Defaulting to 'info'.")

    return logger


# Helper function to load the config file based on the task ID
def load_config(config_path, taskid):
    logger = logging.getLogger(__name__)
    try:
        with open(config_path, 'r') as f:
            config_array = yaml.safe_load(f)
        experiment_key = f"experiment{taskid}"
        return config_array[experiment_key]
    except Exception as e:
        logger.error(f"Impossible to load the config file '{config_path}': {e}", exc_info=True)
        raise


# Helper function to set up argument parsing
def get_parser():
    parser = argparse.ArgumentParser(description='Simple PyTorch Training Script.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--taskid', type=int, default=0, help='Task ID (default: 0)')
    return parser


# Main function that orchestrates the training process
def main(config_uptdate, verbose='info', logger=None, log_wandb=False):
    # Default config
    config = {
        # Dataset size and type
        'num_individuals': -1,  # -1 for all individuals
        'answer_type': 'candidates',

        # Data preprocessing
        'only_full': True,
        'replace_nan': False,
        'clip_values': True,
        'remove_neutral': True,

        # Model architecture
        'encoder_type': 'bert-base-uncased',
        'embedding_dim': 512,
        'projector_hidden_layers': [512, 256],
        'dropout': 0.3,

        # Training parameters
        'val_split': 0.1,
        'split_mode': 'random',
        'epochs': 100,
        'batch_size': 16,
        'lr': 0.001,
        'margin': 0.0,
        'scheduler_patience': 5,
        'scheduler_factor': 0.1,  # Set to 1 to disable the scheduler
        'min_lr': 1e-6,
        'early_stopping_patience': 10,  # Set to -1 to disable early stopping

        # Device selection
        'multigpu': True
    }

    # Manage the logger
    if logger is None:
        if verbose in ['debug', 'info', 'warning', 'error', 'critical']:
            logger = set_logger(__name__, verbose=verbose)
        else:
            logger = set_logger(__name__)
            logger.warning(f"Invalid logging level '{verbose}'. Using default logger.")
    elif not isinstance(logger, logging.Logger):
        logger = set_logger(__name__)
        logger.error("Invalid logger object. Using default logger.")
    elif verbose is not None:
        logger.warning("Both 'verbose' and 'logger' arguments were provided. Using the logger argument.")

    # Update the default config with the new config
    config.update(config_uptdate)

    # Select the device (GPU or CPU)
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb with a specific run name
    if log_wandb:
        datestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{datestr}_{config['encoder_type'].split('/')[-1]}"
        project_name = PROJECT_NAME
        wandb.init(project=project_name, config=config, name=run_name, mode='online')
        logger.info(f"Initialized wandb run '{wandb.run.name}' in project '{project_name}'.")

    # Print relevant information
    logger.info(f"Running the script with the following config:")
    for key, value in config.items():
        logger.info(f"- {key}: {value}")
    logger.info(f"{n_gpus} GPUs detected. Using {device}.")

    # Create the dataset that returns pre-computed embeddings
    encoder = TextEncoder(model_type=config['encoder_type'])
    ds = load_dataset(encoder=encoder, config=config)
    ds_tr, ds_val = split_dataset(ds, val_split=config['val_split'], split_mode=config['split_mode'])

    # Create dataloader
    dl_tr = DataLoader(ds_tr, batch_size=config['batch_size'], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=config['batch_size'], shuffle=False)

    # Create models
    model = AgreeDisagreeModel(num_individuals=len(ds), embedding_dim=config['embedding_dim'],
                               encoder_dim=encoder.output_size, projector_hidden_layers=config['projector_hidden_layers'],
                               dropout=config['dropout'])

    # Move models to device
    if n_gpus > 1 and config['multigpu']:
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Create the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_factor'],
                                                     patience=config['scheduler_patience'], min_lr=config['min_lr'])
    # Define loss function
    loss_fn = AgreeDisagreeLoss(margin=config['margin'])
    #loss_fn = CustomCrossEntropyLoss()
    tracker_tr = MetricTracker()
    tracker_val = MetricTracker()
    early_stopper = EarlyStopping(patience=config['early_stopping_patience'])

    # Train the model
    logger.info("Starting training...")
    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()
        model.train()
        tracker_tr.reset_epoch()

        # TODO incorporate into config
        if epoch == -1:
            # Freeze projector
            logger.info("Freezing projector.")
            if isinstance(model, torch.nn.DataParallel):
                for param in model.module.projector.parameters():
                    param.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.module.parameters()), lr=config['lr'])
            else:
                for param in model.projector.parameters():
                    param.requires_grad = False
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])

        for idx, batch in enumerate(dl_tr):
            # Forward pass
            i, _, e_q, y = batch  # i: indices, e_q: question embeddings, a: answer values
            i = i.to(device)
            e_q = e_q.to(device)
            y = y.to(device)

            # Extract opinions of i and project question embeddings
            outputs = model(i, e_q)

            # Compute loss
            loss = loss_fn(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            with torch.no_grad():
                tracker_tr.update(outputs, y, loss.item() * len(y))

        # Validation phase
        model.eval()
        tracker_val.reset_epoch()
        with torch.no_grad():
            for idx, batch in enumerate(dl_val):
                # Extract batch
                i, _, e_q, y = batch  # i: indices, e_q: question embeddings, a: answer values
                i = i.to(device)
                e_q = e_q.to(device)
                y = y.to(device)

                # Extract opinions of i and project question embeddings
                outputs = model(i, e_q)

                # Compute the metrics
                with torch.no_grad():
                    total_loss = loss_fn(outputs, y).item() * len(y)
                    tracker_val.update(outputs, y, total_loss)

            # Print training and validation loss
            metrics_tr = tracker_tr.get_metrics()
            metrics_val = tracker_val.get_metrics()
        
        # Log the epoch metrics
        t = time.time() - t0
        logger.info(f"Epoch {epoch:3d}/{config['epochs']} ({t:.2f}s) | Train: loss {metrics_tr['loss']:.4f}, acc {metrics_tr['accuracy'] * 100:.2f} | Validation: loss {metrics_val['loss']:.4f}, acc {metrics_val['accuracy'] * 100:.2f} | lr {scheduler.get_last_lr()[0]:.1e}")

        # Log to wandb
        if log_wandb:
            data = {'epoch': epoch,
                    'metrics_tr': metrics_tr,
                    'metrics_val': metrics_val}
            wandb.log(data, commit=True)

        # Check for early stopping
        early_stopper.update(metrics_val['loss'])
        if early_stopper.stop:
            logger.info(f"Early stopping after epoch {epoch}.")
            break

        # Update the learning rate
        scheduler.step(metrics_val['loss'])

    logger.info("Finished Training.")

    # Finish the wandb run
    if log_wandb:
        wandb.finish()

    return {'metrics_tr': metrics_tr,
            'metrics_val': metrics_val,
            'model': model,
            'dataset_tr': ds_tr,
            'dataset_val': ds_val}


if __name__ == "__main__":
    logger = set_logger(__name__, verbose='info')
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config
    taskid = args.taskid
    config_dict = load_config(config_file, taskid)
    _ = main(config_dict, logger=logger, verbose=None, log_wandb=True)
