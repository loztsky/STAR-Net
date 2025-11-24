| Parameter Name | Function/Purpose | Default Value |
| :------------- | :---------------- | :----------- |
| `model` | Specifies the model name to be used. | `"starnet"`, `"mvanet"`, `"mvnet"`, `"tmvanet"` |
| `dataset` | Specifies the dataset name to be used. | `"carrada"`, `"othr"` |
| `process_signal` | Boolean indicating whether to preprocess the original signal. | `true` |
| `add_temp` | Boolean indicating whether to add temporal dimension to input. | `true`, `false` |
| `annot_type` | Type of annotation data, "dense" indicates dense pixel-level annotations (semantic segmentation). | `"dense"` |
| `n_Input` | Number of input views or sequence length. | `1`, `2`, `3` |
| `n_Output` | Number of output views or sequence length. | `1`, `2` |
| `nb_classes` | Number of classes in classification task (including background). | `3`, `4` |
| `nb_input_channels` | Number of input channels per sample (e.g., radar data may have multiple channels). | `5` |
| `hidden_channels` | Number of hidden channels in LSTM modules (STENet specific). | `128` |
| `num_layers` | Number of layers in LSTM modules (STENet specific). | `3` |
| `nb_epochs` | Total number of training epochs. | `5`, `50`, `300`, `400` |
| `batch_size` | Number of samples used in each training iteration. | `4`, `5`, `8` |
| `lr` | Learning rate, controls the magnitude of model parameter updates. | `0.0001` |
| `lr_step` | Frequency of learning rate scheduler adjustments (usually in epochs). | `20` |
| `loss_step` | Interval for recording loss values (in batches). | `100` |
| `val_epoch` | Interval for evaluating the model on validation set (in epochs). | `1`, `5` |
| `viz_step` | Interval for visualization (if enabled) (in batches). | `4000` |
| `torch_seed` | Random seed for PyTorch library to ensure experiment reproducibility. | `40`, `42` |
| `numpy_seed` | Random seed for NumPy library to ensure experiment reproducibility. | `42` |
| `version` | Configuration or model version number for experiment tracking. | `0` |
| `schedular` | Type of learning rate scheduler. | `"exp"`, `"coswarm"` |
| `optimizer` | Type of optimizer for model parameters. | `"Adam"` |
| `device` | Device for model training and inference. | `"cuda:0"` |
| `custom_loss` | Name or combination of custom loss functions to be used. | `"wce_w10sdice"`, `"wce_w10sdice_w5col"` |
| `transformations` | Data augmentation methods applied to input data, e.g., flips. | `"flip"`, `"hflip,vflip"` |
| `norm_type` | Type of data normalization, "tvt" may indicate separate normalization for train/val/test sets. | `"tvt"` |
| `rd_loss_weight` | Weight for Range-Doppler (RD) view related loss (if model outputs multiple views). | `1` |
| `ra_loss_weight` | Weight for Range-Azimuth (RA) view related loss (if model outputs multiple views). | `1` |
| `shuffle` | Boolean indicating whether to shuffle training data at the beginning of each epoch. | `true` |
| `comments` | Comments or descriptive information about the configuration file. | `"Spatio-Temporal Attention Refinement Network (STARNet). Methods: data aug (hflip, vflip) + multi loss + 5 input frames. Model selection: mean of precision. Normalisation: TVT. Loss: wCE + weighted Soft Dice Loss (10 * SDice)."` |

## Model-Specific Parameters

### STARNet (Spatio-Temporal Attention Refinement Network)
- `n_Input`: 1 (RD view only)
- `n_Output`: 1 (RD view only)
- `hidden_channels`: 128 (LSTM hidden channels)
- `num_layers`: 3 (LSTM layers)
- `scheduler`: "coswarm" (Cosine Annealing with Warm Restarts)

### MVANet (Multi-View ASPP Network)
- `n_Input`: 3 (RD, RA, AD views)
- `n_Output`: 2 (RD, RA views)
- `custom_loss`: "wce_w10sdice_w5col" (Weighted Cross Entropy + Weighted Soft Dice + Coherence Loss)
- `rd_loss_weight`: 1
- `ra_loss_weight`: 1

### MVNet (Multi-View Network)
- `n_Input`: 2 (RD, RA views)
- `n_Output`: 2 (RD, RA views)
- `add_temp`: false
- `transformations`: "flip"

### TMVANet (Temporal Multi-View ASPP Network)
- `n_Input`: 3 (RD, RA, AD views)
- `n_Output`: 2 (RD, RA views)
- `add_temp`: true
- Uses 3D convolution for temporal processing
- `custom_loss`: "wce_w10sdice_w5col"

## Loss Function Configurations

### wce_w10sdice
- Weighted Cross Entropy Loss + Weighted Soft Dice Loss (10x weight)

### wce_w10sdice_w5col
- Weighted Cross Entropy Loss + Weighted Soft Dice Loss (10x weight) + Weighted Coherence Loss (5x weight)

## Data Augmentation Options

### flip
- Basic random flipping (horizontal and vertical)

### hflip,vflip
- Separate horizontal and vertical flip controls
- Applied randomly with 50% probability each

## Dataset Configurations

### Carrada Dataset
- Automotive radar dataset
- Contains RD, RA, AD views
- Supports multi-view training

### OTHR Dataset
- Over-the-Horizon Radar dataset
- Primarily supports RD view
- Simpler data structure with train/val/test splits
