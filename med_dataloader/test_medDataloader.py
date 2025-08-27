import torch
import numpy as np
import argparse
import torch
from med_dataloader.dataloader_total import MedicalDataloader

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    train_loader = None
    val_loader = None
    
    if args.dataset == 'smalldata':
        root_path = r"/workspace/datasets/Medical/smalldata"

    elif args.dataset == 'blender':
        root_path = r"/workspace/datasets/Medical/blender-duodenum-5-211126"

    elif args.dataset == 'colon':
        root_path = r"/workspace/datasets/Medical/Colonscopy"


    train_dataset = MedicalDataloader(root_path, type='train', dataset=args.dataset)
    val_dataset = MedicalDataloader(root_path, type='val', dataset=args.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               shuffle=False, num_workers=args.workers, 
                                               pin_memory=True, sampler=None,
                                               worker_init_fn=lambda work_id: np.random.seed(work_id))
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers,
                                             pin_memory=True)

    print("=> data loaders created.")
    return train_loader, val_loader



def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Data Loader Test')
    parser.add_argument('--dataset', type=str, default='colon', help='Dataset to use: smalldata, blender, col')
    parser.add_argument('--max_depth', type=float, default=-1.0, help='Max depth value, -1 means infinity')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers for data loading')

    args = parser.parse_args()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)

    # Check if train_loader and val_loader are created properly
    print("=> Train and validation loaders created successfully.")
    
    # Print the first batch for verification
    print("=> train_loader num:", args.batch_size * len(train_loader))
    print("=> val_loader num:", args.batch_size * len(val_loader))

    maxgt = 0
    if train_loader is not None:
        for i, data in enumerate(train_loader):
            print(f"Train batch rgb_np {i}: {data['image'].shape}, max:{data['image'].max()}, min:{data['image'].min()}")
            print(f"Train batch depth_np {i}: {data['depth'].shape}, max:{data['depth'].max()}, min:{data['depth'].min()}")
            print('----------------------------------')
            if maxgt < data['depth'].max():
                maxgt = data['depth'].max()
            # break  # Only print the first batch to check
    print('train_dataset max gt:', maxgt)
    if val_loader is not None:
        for i, data in enumerate(val_loader):
            print(f"Validation batch rgb_np {i}: {data['image'].shape}")
            print(f"Validation batch depth_np {i}: {data['depth'].shape}")
            break  # Only print the first batch to check

if __name__ == '__main__':
    main()
