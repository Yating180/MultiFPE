# MultiFPE

The official code for the paper _'MultiFPE: Multi-Representation Fusion Network for Facial Palsy Evaluation in Facial Videos'_.
MultiFPE is a novel multi-representation fusion framework for FPE, designed to effectively integrate complementary facial representations for reliable facial palsy evaluation.

<p align="center">
<img src="Pipeline.png" width="100%" />
</p>

## Project Structure

```text
MultiFPE/
├── annotation/                         # Train/test split files, named by dataset and fold
├── ckpt/                               # Pretrained weights
│   ├── ir50.pth
│   └── mobilefacenet_model_best.pth.tar
├── dataloader/                         # Video frame, static frame, landmark loading, and image augmentation
├── models/                             # MultiFPE model architecture
└── main_SCB_RJCMA.py                   # Training and validation entry point
```

## Environment Setup

Install the following dependencies:

```bash
pip install torch==2.0.1 torchvision==0.15.2 timm==0.6.13 numpy==1.26.4 matplotlib==3.7.2 scikit-learn==1.5.1 pillow==9.4.0
```

## Pretrained Weights

Before training, make sure the following files exist under `ckpt/`:

```text
ckpt/ir50.pth
ckpt/mobilefacenet_model_best.pth.tar
```

## Data Preparation

The training script reads the training and test sets from `annotation/<data_set>_train.txt` and `annotation/<data_set>_test.txt`. The current repository contains the following 5-fold splits:

```text
AFLFP_1 ~ AFLFP_5
MEEI_1  ~ MEEI_5
```

## Create Output Directories

The training script writes logs, curve figures, and checkpoints, but it does not create output directories automatically. Create the directories before the first training run.

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path `
  log/model1/MEEI, log/model1/AFLFP, `
  checkpoint/model1/MEEI, checkpoint/model1/AFLFP, `
  best_checkpoint/model1/MEEI, best_checkpoint/model1/AFLFP
```

Linux/macOS:

```bash
mkdir -p log/model1/{MEEI,AFLFP} checkpoint/model1/{MEEI,AFLFP} best_checkpoint/model1/{MEEI,AFLFP}
```

## Single-Fold Training and Validation

Run the training command from the project root directory.

Example: train and validate MEEI fold 5:

```bash
python main_SCB_RJCMA.py --data_set MEEI_5 --epochs 100 -b 16 -j 8 --lr 0.01
```

Example: train and validate AFLFP fold 1:

```bash
python main_SCB_RJCMA.py --data_set AFLFP_1 --epochs 100 -b 16 -j 8 --lr 0.01
```

During training, each epoch performs the following steps:

1. Train on `<data_set>_train.txt`.
2. Validate on `<data_set>_test.txt`.
3. Record Accuracy, F1-score, Precision, and Recall for both the training and validation sets.
4. Save the current checkpoint.
5. If the current validation accuracy is higher, save the best checkpoint and the best confusion matrix.

## 5-Fold Training

Windows PowerShell:

```powershell
foreach ($fold in 1..5) {
  python main_SCB_RJCMA.py --data_set "MEEI_$fold" --epochs 100 -b 16 -j 8 --lr 0.01
}
```

Linux/macOS:

```bash
for fold in 1 2 3 4 5; do
  python main_SCB_RJCMA.py --data_set MEEI_${fold} --epochs 100 -b 16 -j 8 --lr 0.01
done
```

Replace `MEEI` with `AFLFP` to train the 5 folds of the AFLFP dataset.

## Common Parameters

| Parameter | Default      | Description |
| --- |----------| --- |
| `--data_set` | `MEEI_5` | Dataset and fold, for example `MEEI_5` or `AFLFP_1` |
| `--epochs` | `100`    | Total number of training epochs |
| `-b`, `--batch-size` | `16`     | Batch size |
| `-j`, `--workers` | `8`      | Number of DataLoader workers |
| `--lr` | `0.01`   | Initial learning rate |
| `--momentum` | `0.9`    | SGD momentum |
| `--wd` | `1e-4`   | Weight decay |
| `--print-freq` | `4`      | Training/validation log print frequency |
| `--resume` | `None`   | Checkpoint path for resuming training |

