## FlashSparse: Minimizing Computation Redundancy for Fast Sparse Matrix Multiplications on Tensor Cores
**FlashSparse is accepted by PPoPP 2025**. See the Arxiv **[preprint version of the paper](https://arxiv.org/abs/2412.11007)**. FlashSparse significantly reduces the computation redundancy for unstructured sparsity (for SpMM and SDDMM) on Tensor Cores through a **Swap-and-Transpose** mapping strategy.
![FlashSparse](FlashSparseThumbnail.png)

## 1. Clone this project.
```
git clone --recursive https://github.com/ParCIS/FlashSparse.git
```
+ **Requirements**: 
> + `Ubuntu 16.04+`
> + `cmake >= 3.29`
> + `CUDA >= 11.8`
> + one NVIDIA RTX4090 GPU and one H100 PCIe GPU.

## 2. Environment Setup.
Conda environments need to be set up on machines with H100 PCIe and RTX4090 GPUs following the steps below.

### 2.1 Install via Conda.
+ 2.1.1 Install **`conda`** on system. **[(Toturial)](https://docs.anaconda.com/anaconda/install/linux/)**.
+ 2.1.2 Create a **`conda`** environment: 
```
conda create -n env_name python=3.9
```
+ 2.1.3 Install **`PyTorch`** **[(Toturial)](https://pytorch.org/get-started/locally/)**: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
```

## 3. Install **`FlashSparse`**.
```
cd FlashSparse/
bash compile.sh
``` 

## 4. Download datasets. (optional)
```
cd dataset/
The sparse matrices can be downloaded from the SuiteSparse repository, while GNN-related datasets are available on the official websites of DGL and PyG. After downloading, the matrices can be partitioned and preprocessed using FS_Block_gpu.preprocess_gpu_fs(). The preprocessed results are then used for computation with FlashSparse.(total 515 sparse matrices).
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

## 7. Running **FlashSparse** on H100 PCIe and RTX4090 GPUs.
### 7.1 SpMM test
> +  Go to project `eva/kernel/spmm/` directory.
> + `bash ./test_spmm_shell.sh` to run all SpMM experiments. (about 200 minutes)
> + Check the results in `result/FlashSparse/spmm/*.csv`.

### 7.2 SDDMM test
> +  Go to project `eva/kernel/sddmm/` directory.
> + `bash ./test_sddmm_shell.sh` to run all SDDMM experiments. (about 100 minutes)
> + Check the results in `result/FlashSparse/sddmm/*.csv`.

### 7.3 GCN and AGNN tests
> +  Go to project `eva/end2end/gcn/` directory.
> + `python eva_gcn_fs.py` to run GCN experiments.
> + `python eva_gcn_baseline.py` to run GCN experiments.
> + Check the results in `result/FlashSparse/gcn/fs_gcn_128.csv`. (about 5 minutes)
> + Check the results in `result/Baseline/agnn/baseline_gcn_128.csv`. (about 15 minutes)

> +  Go to project `eva/end2end/agnn/` directory.
> + `python eva_agnn_fs.py` to run AGNN experiments.
> + `python eva_agnn_baseline.py` to run AGNN experiments.
> + Check the results in `result/FlashSparse/agnn/fs_agnn_32.csv`. (about 5 minutes)
> + Check the results in `result/Baseline/agnn/baseline_agnn_32.csv`. (about 15 minutes)

## 8. Running **Baselines** on H100 PCIe and RTX4090 GPUs (Optional).
### 8.1 Evaluating **RoDe, Sputnik and cuSPARSE**.
> +  Go to project `Baseline/RoDe/script/` directory.
> + `bash download.sh` to download the same 515 matices in a specific format for RoDe. (optional)
> + `bash test_spmm_shell.sh` to run all SpMM experiments. (about 300 minutes)
> + `bash test_sddmm_shell.sh` to run all SDDMM experiments. (about 300 minutes)
> + Check the results in `result/Baseline/spmm/rode*.csv` and `result/Baseline/sddmm/rode*.csv`.

### 8.2 Evaluating **DTC-SpMM**.
> +  Go to project `Baseline/DTC-SpMM/` directory.
> + `bash test_spmm_shell.sh` to run all SpMM experiments. (about 20 minutes)
> + Check the results in `result/Baseline/spmm/dtc*.csv`.

### 8.3 Evaluating **GNNAdvisor, TCGNN, GE-SpMM**.
> +  Go to project `eva/kernel/spmm/` directory.
> + `bash test_spmm_shell_base.sh` to run all SpMM experiments.  (about 100 minutes)
> + Check the results in `result/Baseline/spmm/base*.csv`.

> +  Go to project `eva/kernel/sddmm/` directory.
> + `bash test_sddmm_shell_base.sh` to run all SDDMM experiments. (about 20 minutes)
> + Check the results in `result/Baseline/sddmm/base*.csv`.

### 8.4 Summarize the results in all baselines.
> +  Go to project `result/Baseline/spmm/` directory.
> + `python summarize.py` to summarize all results.

> +  Go to project `result/Baseline/sddmm/` directory.
> + `python summarize.py` to summarize all results.

## 9 Reproducing the experinmental figures and tables in FlashSparse.
### 9.1 Reproduce the Figure 11 and Table 5. (both on H100 and RTX4090)
> +  Go to project `eva/plot/kernel_spmm/` directory.
> + `python plot_figure11_ac.py` and check the figure in `figure11.png` 
(The plotted figure11.png on H100 corresponds to Figure 11(a) in the paper, and on RTX4090 corresponds to Figure 11(c) in the paper.)
> + `python plot_figure11_bd.py` and check the figure in `figure11_sub.png`.
(The plotted figure11_sub.png on H100 corresponds to Figure 11(b) in the paper, and on RTX4090 corresponds to Figure 11(d) in the paper.)
> + `python profile_table5.py` and check the result in `table5.txt`.
(The profiled table5.txt on H100 corresponds to Table5(left) in the paper, and on RTX4090 corresponds to Table5(right) in the paper.)

### 9.2 Reproduce the Figure 13 and Table 6. (both on H100 and RTX4090)
> +  Go to project `eva/plot/kernel_sddmm/` directory.
> + `python plot_figure13_a.py` and check the figure in `figure13(a).png`.
> + `python plot_figure13_b.py` and check the figure in `figure13(b).png`.
(The plotted figure13(a).png and figure13(b).png on H100 correspond to Figure 13(a)(b) in the paper, and on RTX4090 corresponds to Figure 13(c)(d) in the paper.)
> + `python profile_table6.py` and check the result in `table6.txt`.
(The profiled table6.txt on H100 corresponds to Table6(left) in the paper, and on RTX4090 corresponds to Table6(right) in the paper.)

### 9.3 Reproduce the Figure 12. (only on H100 or RTX4090)
> +  Go to project `eva/plot/ablation/memory/` directory.
> + `python spmm.py` and check the result in `memory_spmm.csv`. (about 20 minutes)
> + `python sddmm.py` and check the result in `memory_sddmm.csv`.  (about 20 minutes)

> + `python plot_spmm.py` and check the figure in `spmm_mem.png`.
> + `python plot_sddmm.py` and check the figure in `sddmm_mem.png`.

### 9.4 Reproduce the Figure 14. (both on H100 and RTX4090)
> +  Go to project `eva/plot/ablation/throughput/` directory.
> + `python plot_spmm.py` and check the figure in `figure14(a).png`.
> + `python plot_sddmm.py` and check the figure result in `figure14(b).png`.
(The plotted figure14(a).png and figure14(b).png on H100 correspond to Figure 14(a)(b) in the paper, and on RTX4090 corresponds to Figure 14(c)(d) in the paper.)

### 9.5 Reproduce the Figure 15. (both on H100 and RTX4090)
> +  Go to project `eva/plot/ablation/access/` directory.
> + `python plot.py` and check the figure in `figure15.png`.
(The plotted figure15.png on H100 correspond to Figure 15(left) in the paper, and on RTX4090 corresponds to Figure 15(right) in the paper.)

### 9.6 Reproduce the Table 7. (only on H100 or RTX4090)
> +  Go to project `eva/plot/ablation/format/` directory.
> + `python format.py` and check the result in `result.csv`.  (about 25 minutes)
> + `python profile.py` and check the output`.

### 9.7 Reproduce the Figure 16. (only on H100 or RTX4090)
> +  Go to project `eva/plot/gcn/` directory.
> + `python plot.py` and check the figure in `figure16_gcn.png`.

> +  Go to project `eva/plot/agnn/` directory.
> + `python plot.py` and check the figure in `figure16_agnn.png`.

### 9.8 Reproduce the Table 8. (only on H100 or RTX4090)
> +  Go to project `eva/accuracy/gcn/` directory.
> + `python eva_gcn.py`. (about 1 minutes)
> +  Check the result in `result/Baseline/gcn/accuracy.csv`.
