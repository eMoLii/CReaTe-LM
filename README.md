# CReaTe-LM

This repository contains the code and resources for constructing and training **CReaTe-LM**, a large language model designed for **clinical reasoning instruction**.

---

## 🔧 Environment Setup

Use the following command to create the conda environment:

```bash
conda env create -f environment.yml
```
---

## 📁 CRID Dataset

We release a de-identified sample of 200 cases at `Datasets/CRID.json` for reproducibility. Full-dataset access is restricted by ethics approval.  Qualified researchers may request access for academic, non-commercial research by emailing (211123120101@zjut.edu.cn).

---

## 🧹 CRID Construction


1. Overview of the dataset construction:

    (/Assets/dataset.png)

2. Navigate to the `CRID_Construction` directory:

    ```bash
    cd CRID_Construction
    ```

3. Run `extraction_generation.py` to convert raw EMRs into CRF-formatted EMRs:

    ```bash
    python extraction_generation.py
    ```

4. Run `dialogGeneration.py` to generate the clinical teaching dialogue dataset `CRID`:

    ```bash
    python dialogGeneration.py
    ```
---

## 🧠 Model Training & Inference

1. Navigate to the `Train` directory:

    ```bash
    cd Code/Train
    ```

2. Run `data2SFT.py` to convert the dialogue data into a format suitable for SFT training:

    ```bash
    python data2SFT.py
    ```

3. Use llama-factory for effient SFT training. Please refer to its official guide (https://github.com/hiyouga/LLaMA-Factory). The config file is:

    ```
    SFT_args.yaml
    ```

4. Use `llama-factory` for DPO training. The configuration file is:

    ```
    DPO_args.yaml
    ```

    The training data is located at:

    ```
    Datasets/DPO_data.json
    ```

5. For efficient inference, use vLLM. Please refer to the official vLLM documentation for setup and usage (https://github.com/vllm-project/vllm).

---

## 📊 Model Evaluation

1. Navigate to the `Evaluation` directory:

    ```bash
    cd Code/Evaluation
    ```

2. Run `eval.py` to obtain model evaluation results:

    ```bash
    python eval.py
    ```

---

## 📜 License
This repository is licensed under the Apache-2.0 License(https://github.com/eMoLii/CReaTe-LM/blob/main/LICENSE).
