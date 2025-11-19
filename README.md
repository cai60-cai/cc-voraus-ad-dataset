
You are an expert machine-learning code generator and research engineer.
Now you will implement an entire benchmark project for:

**“Variable-Cardinality, Missing-Feature Robust Models for Robot Sensor Time-Series Anomaly Detection”**

The goal is to automatically generate a complete multi-method codebase with
**8 separate model implementations**, each following the original papers,
with full training loops, evaluation, GPU support, progress bars,
and ability to run inference after each epoch.

---

# 🧱 **GLOBAL PROJECT REQUIREMENTS (APPLY THIS TO ALL FILES AND METHODS)**

### **1. Data & Task**

* Each sample is a full pick-and-place cycle from an industrial robot.
* Input shape: `[T, S]`, where

  * `T ≈ 1000–2000` time steps
  * `S = 130` sensor channels
* Feature-level missing sensors exist: some channels are zero for entire sample.
* Missing is **not imputed**—we use a mask.
* Training = **normal-only one-class anomaly detection**.
* Evaluation = normal + 12 mechanical anomaly classes.
* After every epoch: run inference on test set (ROC-AUC + PR-AUC).

### **2. Data interface (shared across all models)**

Create `data/dataset.py` that yields:

```
{
  "x": tensor [T, S],
  "mask": tensor [T, S],   # 1 for observed, 0 for missing
  "label": int (0 or 1),   # used only in eval
}
```

### **3. Folder structure**

Generate this entire folder structure with placeholder files as needed:

```
project_root/
  data/
    dataset.py

  common/
    config.py
    train_loop.py
    eval.py
    utils.py

  methods/
    set_transformer/
      model.py
      train.py
      config_set_transformer.py

    naim/
      model.py
      train.py
      config_naim.py

    deepsets/
      model.py
      train.py
      config_deepsets.py

    ft_transformer/
      model.py
      train.py
      config_ft_transformer.py

    neumiss/
      model.py
      train.py
      config_neumiss.py

    dropout_baseline/
      model.py
      train.py
      config_dropout.py

    mlp_impute/
      model.py
      imputation.py
      train.py
      config_mlp_impute.py

    tabpfn/
      run_tabpfn.py
```

### **4. Training loop**

Implement `common/train_loop.py` with:

* PyTorch
* GPU support
* **no `torch.compile`**
* tqdm progress bars
* mixed precision optional
* after each epoch:

  * run evaluation loop
  * compute ROC-AUC + PR-AUC via sklearn
  * save checkpoint
* `Method` interface:

```
class Method(nn.Module):
    def forward(self, x, mask) -> scores
    def training_step(self, batch) -> loss
    def eval_step(self, batch) -> (scores, labels)
```

### **5. Evaluation**

`common/eval.py` must provide:

* compute anomaly scores
* ROC-AUC, PR-AUC
* a simple function:

```
evaluate(model, dataloader) -> {"auc":..., "pr":...}
```

### **6. Hyperparameters**

* All hyperparameters in `config_xxx.py` (dataclasses).
* Use the paper-recommended defaults.

---

# 📚 **NOW IMPLEMENT ALL 8 METHODS**

Each method must **faithfully follow the referenced paper**.
Each must consume `(x, mask)` and output anomaly scores.

---

# 1️⃣ **SET TRANSFORMER**

(Paper: Lee et al., ICML 2019)

### Requirements:

* Per-sensor encoder φ:

  * shared temporal encoder
  * 1D TCN over time input `[T,2]` (value + mask)
  * outputs `z_s ∈ R^{d_phi}`
* Add sensor-ID embedding
* Build set `Z = { z_s | mask[:,s].sum()>0 }`
* L layers of SAB (Set Attention Block)
* One PMA seed → global embedding
* Deep SVDD head: distance to learned center

Files to write:

```
methods/set_transformer/model.py
methods/set_transformer/train.py
methods/set_transformer/config_set_transformer.py
```

---

# 2️⃣ **NAIM (Not Another Imputation Method)**

(Paper: NAIM, arXiv:2407.11540)

### Requirements:

* One token per sensor channel
* **Masked self-attention**: attention logits for missing sensors = -∞
* Transformer encoder layers
* CLS pooling or PMA
* One-class output

Files:

```
methods/naim/model.py
methods/naim/train.py
methods/naim/config_naim.py
```

---

# 3️⃣ **DEEP SETS**

(Zaheer et al., NeurIPS 2017)

### Requirements:

* Shared φ (MLP or temporal encoder)
* Permutation-invariant sum pooling
* ρ MLP head → anomaly score
* Minimalistic, lightweight

Files:

```
methods/deepsets/model.py
methods/deepsets/train.py
methods/deepsets/config_deepsets.py
```

---

# 4️⃣ **FT-TRANSFORMER**

(Gorishniy et al., FT-Transformer)

### Requirements:

* Feature tokenizer:

  * continuous features → linear layer
  * add feature-ID embedding
* Transformer encoder
* CLS pooling
* One-class head

Files:

```
methods/ft_transformer/model.py
methods/ft_transformer/train.py
methods/ft_transformer/config_ft_transformer.py
```

---

# 5️⃣ **NEUMISS**

(NeuMiss: Le Morvan et al., NeurIPS 2020)

### Requirements:

* NeuMiss layer with Neumann series missing-aware linear operator
* Accepts (values, mask)
* Stack NeuMiss layers + nonlinearity
* Final embedding → one-class head
* Must follow paper exactly

Files:

```
methods/neumiss/model.py
methods/neumiss/train.py
methods/neumiss/config_neumiss.py
```

---

# 6️⃣ **FEATURE DROPOUT BASELINE**

(Training technique)

### Requirements:

* Simple MLP or temporal encoder
* During training: randomly drop features (structured + unstructured)
* During inference: feed actual missing mask
* One-class loss

Files:

```
methods/dropout_baseline/model.py
methods/dropout_baseline/train.py
methods/dropout_baseline/config_dropout.py
```

---

# 7️⃣ **MLP + IMPUTATION BASELINE**

(Weak baseline; for comparison only)

### Requirements:

* Simple imputation: mean or KNN
* Convert missing → imputed value
* Feed 130-dim vector to MLP
* One-class head

Files:

```
methods/mlp_impute/model.py
methods/mlp_impute/imputation.py
methods/mlp_impute/train.py
methods/mlp_impute/config_mlp_impute.py
```

---

# 8️⃣ **TABPFN / TABICL WRAPPER**

(Context learning for tabular)

### Requirements:

* Provide wrapper calling existing TabPFN implementation
* Subsample features/time
* Get anomaly probability / score

Files:

```
methods/tabpfn/run_tabpfn.py
```

---

# 🧩 **ADDITIONAL CODING RULES**

### All code must:

* Use PyTorch
* Move tensors to GPU (`cuda`) when available
* Use tqdm for progress bars
* Use clean modular functions
* Have thorough comments referring to the paper sections
* Not use `torch.compile`
* Run inference after every epoch
* Save checkpoints
* Be deterministic with a `seed_everything()` util

---

# 🚀 **NOW GENERATE THE ENTIRE PROJECT**

Create **all folders, all scripts, all configs, and all implementations**.
Fill in real PyTorch code for all models, training loops, and evaluation.

Produce the code in a clean and organized way.
You may output file-by-file.

---


