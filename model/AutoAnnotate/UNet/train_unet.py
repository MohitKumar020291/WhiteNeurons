import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import choice

from .dataset_loading import load_train_data
from .function_blocks import UNet
from utils import read_yaml_file


def return_org_image_and_label_only(datas):
    images = []
    labels = []
    for data in datas:
        images.append(data[1])  # org image: [H, W, C]
        labels.append(data[3])  # accumulated mask: [H, W, 1]

    images = torch.tensor(np.stack(images))  # [N, H, W, C]
    labels = torch.tensor(np.squeeze(np.stack(labels), axis=-1))  # [N, H, W]

    assert (images.shape[:-1] == labels.shape)

    # For Conv2D
    return images.permute(0, 3, 1, 2).float(), labels.unsqueeze(1).float()


@torch.no_grad()
def validate_one_epoch(model, criterion, val_dataset, batch_size, cat_id, device):
    model.eval()
    total_loss = 0.0
    N = len(val_dataset)
    batches = [choice(N, size=batch_size, replace=False).tolist()
               for _ in range(max(1, N // batch_size))]

    for batch in batches:
        batch.append(cat_id)  # maintain interface
        datas = val_dataset[batch]
        images, labels = return_org_image_and_label_only(datas)
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        assert outputs.shape == labels.shape, f"[VAL] Shape mismatch: {outputs.shape} vs {labels.shape}"
        loss = criterion(outputs, labels)

        total_loss += loss.item()

    return total_loss / len(batches)

def train_one_epoch(model, criterion, optimizer, dataset, batch_size, repeat_batches, cat_id, device):
    model.train()
    epoch_loss = 0.0
    N = len(dataset)
    batches = [choice(N, size=batch_size, replace=False).tolist()
               for _ in range((N // batch_size) * repeat_batches)]

    for batch in batches:
        batch.append(cat_id)
        datas = dataset[batch]
        images, labels = return_org_image_and_label_only(datas)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        assert outputs.shape == labels.shape, f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}"
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(batches)


def save_checkpoint(model, optimizer, epoch, loss, save_path='checkpoint.pth'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)


def main():
    current_directory = os.path.dirname(__file__)
    train_config_path = os.path.join(current_directory, 'unet_config.yaml')
    train_config = read_yaml_file(train_config_path)

    epochs = train_config['epochs']
    batch_size = train_config['bs']
    repeat_batches = train_config['repeat_batches']
    lr = train_config.get('lr', 1e-3)
    checkpoint_dir = train_config.get('checkpoint_dir', 'checkpoints')
    checkpoint_dir = os.path.join(current_directory, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU: cuda\n")
    else:
        device = torch.device("cpu")
        print("Training on CPU\n")

    dataset = load_train_data()
    val_data = load_train_data(type_='valid')
    seg_image = dataset[0]
    cats = seg_image.cats
    print("Sample Image ID (SANITY):", seg_image.image_id)

    models = {cat: UNet().to(device) for cat in cats}
    criterions = {cat: nn.BCEWithLogitsLoss() for cat in cats}
    optimizers = {cat: optim.Adam(models[cat].parameters(), lr=lr) for cat in cats}

    history = dict()
    
    for cat_id, cat in enumerate(cats):
        best_loss = float('inf')
        model = models[cat]
        criterion = criterions[cat]
        optimizer = optimizers[cat]

        print(f"\nTraining for category: {cat} (ID: {cat_id})")

        history[f"cat_{cat_id}"] = dict()

        for epoch in range(epochs):
            history[f"cat_{cat_id}"][epoch] = dict()

            avg_loss = train_one_epoch(model, criterion, optimizer, dataset, batch_size, repeat_batches, cat_id, device)
            val_loss = validate_one_epoch(model, criterion, val_data, batch_size, cat_id, device)

            history[f"cat_{cat_id}"][epoch]['train_loss'] = avg_loss
            history[f"cat_{cat_id}"][epoch]['val_loss'] = val_loss

            # Get overwrite but prevents from sudden training crashes
            with open(f"{checkpoint_dir}/training_history_cat{cat_id}.json", 'w') as f:
                json.dump(history[f'cat_{cat_id}'], f, indent=4)

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = f'{checkpoint_dir}/unet_cat{cat_id}_epoch{epoch+1}.pth'
                save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
                print(f"Checkpoint saved for category '{cat}' at {save_path}")
        
        final_path = f'{checkpoint_dir}/unet_cat{cat_id}_final.pth'
        save_checkpoint(model, optimizer, epoch, avg_loss, final_path)
        print(f"Final checkpoint saved for category '{cat}' at {final_path}")

        print(f"\nFinished training category '{cat}' | Best Loss: {best_loss:.4f}\n{'-'*50}")

    with open(f"{checkpoint_dir}/training_history_all.json", 'w') as f:
        json.dump(history, f, indent=4)
        print(f"Saved Global History at {checkpoint_dir}/training_history_all.json")


    print("Training completed.")


if __name__ == "__main__":
    # from pipeline.loadAnnotedData.helper import show_image
    # val_data = load_train_data(type_='valid')
    main()
