```
NAO_Pytorch/NAO_V1/exp/search_cifar10_big
    ├── arch_embeddings.cifar10.pt
    ├── data_embeddings.cifar10.resnet50.pt
    ├── targets.cifar10.resnet50.pt
    ├── arch_losses.cifar10.pt
    ├── arch_list.pkl
    ├── nao.final
    └── train.log
```

```{python}
class CrossProductInMemoryDataset(Dataset):
    def __init__(self, arch_embeddings_file, data_embeddings_file, targets_file, loss_file):
        self.labels = torch.load(targets_file).long() # Data_Index -> Labels
        self.data_embeddings = torch.load(data_embeddings_file) # Data_Index -> Data Embedding
        self.arch_embeddings = torch.load(arch_embeddings_file) # Arch_Index -> Arch Embedding
        self.losses = torch.load(loss_file) # Arch_Index x Data_Index -> Arch Embedding
        self.arch_list = pickle.load('arch_list.pkl') # Arch Index -> Arch Structure 
        self.num_data, self.data_dim = self.data_embeddings.shape
        self.num_archs, self.arch_dim = self.arch_embeddings.shape
```

Choose 1 in every 500 arch list from arch_list
- Choose indices = [200, 700, ... 2700]
- architectues = arch_list[indices]
- for each arch in architectures: python train_cifar.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="$fixed_arc" \
  --use_aux_head \
  --cutout_size=16 | tee -a $OUTPUT_DIR/train.log

Compare elementwise loss with Logfiles ('arch_losses.cifar10.pt')