# PMDSynergy

PMDSynergy is a pre-training based multi-dimensional fusion model for drug synergy prediction.

Currently, this version supports only the Drugcomb and DrugcombDB datasets. Future versions will include support for custom datasets.

To run PMDSynergy, you need to have CUDA, the GPU version of PyTorch, and torch-geometric installed. Users should install the latest version of these dependencies based on their local configuration by referring to the official websites.

Before running, you must select a dataset. In the `divide.ipynb` file, choose a scenario to perform dataset splitting, resulting in `train_data.csv`, `val_data.csv`, and `test_data.csv`.

Then, copy these three dataset files along with the following three files into the `data/` directory:

1. **MOSA Cell Line Encoding**: `cell_line_latent_values.npy`  
   Cai Z, Apolinário S, Baião A R, et al. Synthetic augmentation of cancer cell line multi-omic datasets using unsupervised deep learning. *Nature Communications*, 2024, 15(1): 10390.

2. **ChemBEARa Drug Encoding**: `ChemBERTa_smiles_embeddings.npy`  
   Chithrananda S, Grand G, Ramsundar B. ChemBERTa: large-scale self-supervised pretraining for molecular property prediction. *arXiv preprint arXiv:2010.09885*, 2020.

3. **GeminiMol Drug Encoding**: `geminimol_latent_dict.npy`

Finally, run the model by executing the following command in the root directory:

```bash
python main.py
