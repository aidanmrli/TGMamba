import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch
import os
import json
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import LightGTMamba
import argparse
from pytorch_lightning.loggers import WandbLogger
from data import DODHDataModule, TUHZDataModule, BCIchaDataModule, BCIchaDataset, MAMEMDataModule, MAMEMDataset

DODH_RAW_DATA_DIR='/home/amli/dreem-learning-open/data/h5/dodh/'
DODH_PROCESSED_DATA_DIR='/home/amli/TGMamba/data/'
TUHZ_DATA_DIR='/home/amli/TGMamba/data/tuhz/processed_dataset/old'
BCICHA_DATA_DIR='/home/amli/TGMamba/data/BCIcha/'
MAMEM_DATA_DIR='/home/amli/MAtt/data/MAMEM/'


def main(args):
    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    
    
    # TODO: instead of loading the train dataset that way, use this datamodule
    project_name_suffix = ""
    stopping_callback_metric = ""
    if args.dataset == 'dodh':
        datamodule = DODHDataModule(
                preprocessed_data_dir=DODH_PROCESSED_DATA_DIR, 
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                balanced_sampling=True,  # default True
                use_class_weight=False,  # default False
                pin_memory=True,
            )
        stopping_callback_metric = "val/macro_f1"
        project_name_suffix = "_dodh"
        input_dim = 129 if args.dataset_has_fft else 1
    elif args.dataset == 'tuhz':
        datamodule = TUHZDataModule(
            data_dir=TUHZ_DATA_DIR,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            dataset_has_fft=args.dataset_has_fft,
        )
        stopping_callback_metric = "val/auroc"
        input_dim = 100 if args.dataset_has_fft else 1
        project_name_suffix = "_tuhz"
    elif args.dataset == 'bcicha':
        SUBJECT_LIST = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
        datamodule = BCIchaDataModule(
            data_dir=BCICHA_DATA_DIR,
            subject=args.subject,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            dataset_has_fft = args.dataset_has_fft
        )
        stopping_callback_metric = "val/auroc"
        input_dim = 11 if args.dataset_has_fft else 1
        project_name_suffix = "_bcicha"
    elif args.dataset == 'mamem':
        SUBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        datamodule = MAMEMDataModule(
            data_dir=MAMEM_DATA_DIR,
            subject=args.subject,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            # dataset_has_fft = args.dataset_has_fft
        )
        stopping_callback_metric = "val/accuracy"
        input_dim = 1
        project_name_suffix = "_mamem"
        
    # Prepare the data
    datamodule.prepare_data()
    datamodule.setup()
    print("Model parameters: ", args.state_expansion_factor, args.seq_pool_type, args.vertex_pool_type, args.conv_type, args.model_dim, args.local_conv_width, args.num_tgmamba_layers)
    # print("Input dim: ", input_dim)
    
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model = LightGTMamba.load_from_checkpoint(args.load_model, 
                                                  dataset=args.dataset, 
                            conv_type=args.conv_type.lower(), 
                            seq_pool_type=args.seq_pool_type, 
                            vertex_pool_type=args.vertex_pool_type,
                            input_dim=input_dim, 
                            d_model=args.model_dim, 
                            d_state=args.state_expansion_factor, 
                            d_conv=args.local_conv_width,
                            num_tgmamba_layers=args.num_tgmamba_layers, 
                            optimizer_name=args.optimizer_name,
                            lr=args.lr_init,
                            weight_decay=args.weight_decay,
                            rmsnorm=args.rmsnorm,
                            edge_learner_attention=True,  # args.edge_learner_attention,
                            attn_threshold=args.attn_threshold,
                            edge_learner_time_varying=False,
                                                  )
    else:
        model = LightGTMamba(dataset=args.dataset, 
                            conv_type=args.conv_type.lower(), 
                            seq_pool_type=args.seq_pool_type, 
                            vertex_pool_type=args.vertex_pool_type,
                            input_dim=input_dim, 
                            d_model=args.model_dim, 
                            d_state=args.state_expansion_factor, 
                            d_conv=args.local_conv_width,
                            num_tgmamba_layers=args.num_tgmamba_layers,
                            gconv_after_all_layers=args.gconv_after_all_layers, 
                            optimizer_name=args.optimizer_name,
                            lr=args.lr_init,
                            weight_decay=args.weight_decay,
                            init_skip_param=args.init_skip_param,
                            num_epochs=args.num_epochs,
                            rmsnorm=True,
                            edge_learner_attention=args.edge_learner_attention,
                            attn_threshold=args.attn_threshold,
                            attn_softmax_temp=args.attn_softmax_temp,
                            edge_learner_time_varying=args.edge_learner_time_varying,
                            edge_learner_layers=args.edge_learner_layers,)

    # Callbacks
    checkpoint_filename = f"{args.dataset}-subject{args.subject}-depth-{args.num_tgmamba_layers}-state-{args.state_expansion_factor}_conv-{args.conv_type}_seq-{args.seq_pool_type}_vp-{args.vertex_pool_type}_fft-{str(args.dataset_has_fft)}_{{epoch:02d}}"
    checkpoint_callback = ModelCheckpoint(
        monitor=stopping_callback_metric,
        mode="max",
        dirpath=args.save_dir,
        filename=checkpoint_filename,
        save_last=True,
        save_top_k=1,  # Save top 3 models
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor=stopping_callback_metric, mode="max", patience=args.patience
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    project_name = "tgmamba" + project_name_suffix
    if args.dataset == 'bcicha':
        wandb_logger = WandbLogger(project=project_name, log_model="all", config=vars(args), tags=[f"subject_{args.subject}"])
    else:
        wandb_logger = WandbLogger(project=project_name, log_model="all", config=vars(args))
        
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=wandb_logger,
        devices=args.gpu_id,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    if not args.load_model:
        trainer.fit(model, datamodule=datamodule)
        print("Training complete.")
        print("Testing best model...")
        best_results = trainer.test(
            datamodule=datamodule,
            ckpt_path="best",
        )
    else:
        print("Testing loaded model...")
        best_results = trainer.test(model, datamodule=datamodule)

    # Print or process the results as needed
    print("Best model results:", best_results)

    # # Save ROC curve data
    # from scipy.io import savemat
    # mdic = {"roc_fpr": fpr, "roc_tpr": tpr}
    # savemat("TGMamba/results/test_roc.mat", mdic)
# /home/amli/TGMamba/results/template/depth-1-state-32_conv-graphconv_seq-mean_vp-max_fft-True_36.ckpt
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGMamba Training Script")
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='TGMamba/results')
    parser.add_argument('--dataset', type=str, default='dodh')
    parser.add_argument('--subject', type=int, default=2)
    parser.add_argument('--dataset_has_fft', action='store_true', help="Enable FFT-processed data in the model")
    parser.add_argument('--conv_type', type=str, default='graphconv')
    parser.add_argument('--seq_pool_type', type=str, default='mean')
    parser.add_argument('--vertex_pool_type', type=str, default='max')
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--state_expansion_factor', type=int, default=32)
    parser.add_argument('--local_conv_width', type=int, default=4)
    parser.add_argument('--num_tgmamba_layers', type=int, default=1)
    parser.add_argument('--gconv_after_all_layers', action='store_true', help="a single GConv layer after all TGMamba layers")
    parser.add_argument('--rmsnorm', action='store_true', help="Enable RMSNorm in the model")
    parser.add_argument('--edge_learner_layers', type=int, default=1)
    parser.add_argument('--edge_learner_attention', action='store_true', help="Enable attention in the edge learner")
    parser.add_argument('--edge_learner_time_varying', action='store_true', help="Enable time-varying edge weights in the edge learner")
    parser.add_argument('--attn_threshold', type=float, default=0.2)
    parser.add_argument('--attn_softmax_temp', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--init_skip_param', type=float, default=0.25)
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--load_model', type=str, default=None, help="Path to a saved model checkpoint to load and test")
    
    args = parser.parse_args()
    main(args)