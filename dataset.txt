# The sparse matrices can be downloaded from the SuiteSparse repository, while GNN-related datasets are available on the official websites of DGL and PyG. 
# After downloading, the matrices can be transformed from .tar.gz to .npz format using the following script.
# The the .npz matrices can be partitioned and preprocessed using FS_Block_gpu.preprocess_gpu_fs(). You can refer this function in FlashSparse/eva/kernel/spmm/fs_fp16/mdataset2.py as an example. 
# The preprocessed results are then used for computation with FlashSparse.
import os
import tarfile
import shutil
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import *
# Define source and target directories
source_dir = './suitesparse' # Path to the directory containing .tar.gz files
target_dir = './temp/sp'  # Target path for moving extracted files


for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.tar.gz'):
            file_path = os.path.join(root, file)
            
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=root)
                print(f"Extracted {file} to {root}")
            
            # Move the extracted file to the target directory
            folder_name = os.path.splitext(file)[0]
            folder_name = os.path.splitext(folder_name)[0]
            extracted_folder = os.path.join(root, folder_name)
            src = extracted_folder + '/' + folder_name + '.mtx'
        
            shutil.copy2(src, target_dir)
            # Delete the extracted folder
            shutil.rmtree(extracted_folder)
            print(f"Deleted {extracted_folder}")
            
            #Read the .mtx file
            file_path =  target_dir + '/' + folder_name + '.mtx'
            src_li1 = []
            dst_li1 = []
            with open(file_path, 'r') as file:
                # Skip comment lines
                for line in file:
                    if line.startswith('%'):
                        continue
                    else:
                        # Read header information
                        head = line.split()
                        break  # Exit after reading the first non-comment line
                    
                # Read the file content line by line
                for line in file:
                    # Remove newline and split by space
                    num= line.strip().split(' ')
                    src_li1.append(int(num[0]))
                    dst_li1.append(int(num[1]))

                num_nodes_src_ = int(head[0])+1
                num_nodes_dst_ = int(head[1])+1
                # max_node = max(max(src_li1), max(dst_li1))
                # num_nodes = max(num_nodes_src_, num_nodes_dst_, max_node)
                num_edges_ = int(head[2])
                edge_index = np.stack([src_li1, dst_li1])
                adj = sp.coo_matrix((np.ones(len(src_li1)), edge_index),
                            shape=(num_nodes_src_, num_nodes_dst_),
                            dtype=np.float32)           
                coo_mat = coo_matrix(adj)

                np.savez('./sp_matrix/' + folder_name + '.npz', 
                        num_nodes_src = num_nodes_src_,
                        num_nodes_dst = num_nodes_dst_,
                        num_edges = num_edges_,
                        src_li = coo_mat.row,
                        dst_li = coo_mat.col)
                print(folder_name + ' is success.')
print("Extraction and moving complete.")
