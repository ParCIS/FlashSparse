# FlashSparse Artifact for PPoPP'25.
## 1. Clone this project.
```
git clone --recursive git https://github.com/ParCIS/FlashSparse.git
```
+ **Requirements**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 11.0` and `nvcc >= 11.0`
> + NVIDIA GPU with `RTX4090 and H100 GPUs`.

## 2. Environment Setup.

### 2.1 Install via Conda.
+ 2.1.1 Install **`conda`** on system. **[(Toturial)](https://docs.anaconda.com/anaconda/install/linux/)**.
+ 2.1.2 Create a **`conda`** environment: 
```
conda create -n env_name python=3.9
```
+ 2.1.3 Install **`Pytorch`** **[(Toturial)](https://pytorch.org/get-started/locally/)**: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
```

## 3. Install **`FlashSparse`**.
```
cd FlashSparse/
bash comple.sh
``` 
to install the TCGNN_conv modules with Pytorch binding. 
**Note that this step is required for both Docker and Conda setup.**


## 4. Download datasets.
Get the preprocessed datasets (total 515 sparse matrices).
```
cd dataset/
python prepare.py
``` 

## 5. Install Sparse-Kernel Baselines.
### Install **`RoDe, Sputnik, GNNAdvisor, GE-SpMM, cuSPARSE, DTC-SpMM and TC-GNN`**: 
```
cd Baseline/
bash comple.sh
```

## 6. Install GNNs-Framework Baselines.
### 6.1 Install **`Deep Graph Library (DGL)`** **[(Toturial)](https://www.dgl.ai/pages/start.html)**: 
```
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
```

### 6.2 Install **`Pytorch-Geometric (PyG)`** **[(Toturial)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)**: 
```      
    pip install torch_geometric
```

## 7. Running **FlashSparse**.
### 7.1 SpMM test
> +  Go to project `eva/kernel/spmm/` directory.
> + `bash ./test_spmm_shell.sh` to run all SpMM experiments.
> + Check the results in `result/FlashSparse/spmm/*.csv`.

### 7.2 SDDDMM test
> +  Go to project `eva/kernel/sddmm/` directory.
> + `bash ./test_sddmm_shell.sh` to run all SpMM experiments.
> + Check the results in `result/FlashSparse/sddmm/*.csv`.

### 7.3 GCN and AGNN tests
> +  Go to project `eva/end2end/gcn/` directory.
> + `bash eva_gcn_fs.py` to run GCN experiments.
> + `bash eva_gvn_baseline.py` to run GCN experiments.
> + Check the results in `result/FlashSparse/gcn/fs_gcn_128.csv`.
> + Check the results in `result/Baseline/agnn/baseline_gcn_128.csv`.

> +  Go to project `eva/end2end/agnn/` directory.
> + `bash eva_agnn_fs.py` to run GCN experiments.
> + `bash eva_agnn_baseline.py` to run GCN experiments.
> + Check the results in `result/FlashSparse/agnn/fs_agnn_32.csv`.
> + Check the results in `result/Baseline/agnn/baseline_agnn_32.csv`.

## 8. Running **Baselines**.
### 8.1 Evaluating **RoDe, Sputnik and cuSPARSE**.
> +  Go to project `Baseline/RoDe/script/` directory.
> + `bash download.sh` to download and preporcess the 515 matices.
> + `bash test_spmm_shell.sh` to run all SpMM experiments.
> + `bash test_sddmm_shell.sh` to run all SDDMM experiments.
> + Check the results in `result/Baseline/spmm/rode*.csv` and `result/Baseline/sddmm/rode*.csv`.

### 8.2 Evaluating **DTC-SpMM**.
> +  Go to project `Baseline/DTC-SpMM/` directory.
> + `bash test_spmm_shell.sh` to run all SpMM experiments.
> + Check the results in `result/Baseline/spmm/dtc*.csv`.

### 8.3 Evaluating **TCGNN, cuSPARSE, GE-SpMM**.
> +  Go to project `eva/kernel/spmm/` directory.
> + `bash test_spmm_shell_base.sh` to run all SpMM experiments.
> + Check the results in `result/Baseline/spmm/base*.csv`.


### 8.4 Summarize the results in all baselines.
> +  Go to project `result/Baseline/spmm/` directory.
> + `python summarize.py` to summarize all results.

> +  Go to project `result/Baseline/sddmm/` directory.
> + `python summarize.py` to summarize all results.

## 9 Reproducing the experinmental figures and tables in FlashSparse.
### 9.1 Reproduce the Figure 11 and Table 5.
> +  Go to project `eva/plot/kernel_spmm/` directory.
> + `python plot_figure11_ac.py` and check the figure in `figure11(a)(c).png`.
> + `python plot_figure11_bd.py` and check the figure in `figure11(b)(d).png`.
> + `python profile_table5.py` and check the result in `table5.txt`.

### 9.2 Reproduce the Figure 13 and Table 6.
> +  Go to project `eva/plot/kernel_sddmm/` directory.
> + `python plot_figure13_a.py` and check the figure in `figure13(a).png`.
> + `python plot_figure13_b.py` and check the figure in `figure13(b).png`.
> + `python profile_table6.py` and check the result in `table6.txt`.

### 9.3 Reproduce the Figure 12.
> +  Go to project `eva/plot/ablation/memory/` directory.
> + `python spmm.py` and check the result in `memory_spmm.csv`.
> + `python sddmm.py` and check the result in `memory_sddmm.csv.`

> + `python plot_spmm.py` and check the figure in `spmm_mem.png`.
> + `python plot_sddmm.py` and check the figure in `sddmm_mem.png`.

### 9.4 Reproduce the Figure 14.
> +  Go to project `eva/plot/ablation/throughput/` directory.
> + `python plot_spmm.py` and check the figure in `figure14(a).png`.
> + `python plot_sddmm.py` and check the refiguresult in `figure14(b).pmg`.

### 9.5 Reproduce the Figure 15.
> +  Go to project `eva/plot/ablation/access/` directory.
> + `python plot.py` and check the figure in `figure16.png`.

### 9.6 Reproduce the Table 7.
> +  Go to project `eva/plot/ablation/format/` directory.
> + `python format.py` and check the result in `result.csv`.
> + `python profile.py` and check the output`.

### 9.7 Reproduce the Figure 16.
> +  Go to project `eva/plot/gcn/` directory.
> + `python plot.py` and check the figure in `figure16_gcn.png`.

> +  Go to project `eva/plot/agnn/` directory.
> + `python plot.py` and check the figure in `figure16_agnn.png`.

### 9.8 Reproduce the Table 8.
> +  Go to project `result/Baseline/gcn/` directory.
> +  Check the result in `accuracy.csv`.
