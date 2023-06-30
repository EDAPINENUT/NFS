# Nonequispaced Neural Fourier Solver for PDEs (NFS)


### Requirements
* torch>=1.8.0 
* tqdm

Dependency can be installed using the following command:
```bash
conda env create --file nfs_env.yaml
conda activate nfs_env
```

### Preparing the datasets
Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1RthMz7xSItlCtTn9FRk9fxckl8z6HDpK?usp=sharing), including Burgers equation, KdV equation and NS equation. Copy them to `./datasets/`.

### Training the Model

Run the following commands to train the model.

```bash
#  Training the Model
python nfs_ns_neq.py # for nonequispaced NS equation
python nfs_ns_eq.py # for equispaced NS equation
```

For other models in our framework, see more examples in `./run/`.
