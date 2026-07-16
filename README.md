# AI for Radiologic Image Tasks: Classification, Detection, and Segmentation


Companion notebooks for the manuscript, *AI for Radiologic Image Tasks: Classification, Detection, and Segmentation*. The manuscript explains the concepts, annotation requirements, evaluation principles, and limitations of each task. These notebooks provide the hands-on counterpart: reproducible, small-scale examples that take the reader from the public datasets to preprocessing, model training, held-out evaluation, and qualitative review.

They are educational workflows, not clinical decision-support systems or deployment templates. In particular, the small public datasets and simplified splits are useful for learning, but do not replace patient-level evaluation, external validation, calibration, regulatory review, or clinical workflow integration.

## Notebooks

| Task | Notebook | Open in Colab |
| :--- | :--- | :---: |
| Classification | [`classification.ipynb`](notebooks/classification.ipynb) | [Open in Colab](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/classification.ipynb) |
| Detection | [`detection_yolo.ipynb`](notebooks/detection_yolo.ipynb) | [Open in Colab](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/detection_yolo.ipynb) |
| Segmentation | [`segmentation.ipynb`](notebooks/segmentation.ipynb) | [Open in Colab](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/segmentation.ipynb) |

- **Classification:** train on Shenzhen TB labels and compare the held-out Shenzhen test set with the independent Montgomery cohort.
- **Detection:** convert Montgomery left- and right-lung masks to boxes, then fine-tune a pretrained YOLO detector for the single `lung` class.
- **Segmentation:** combine the Montgomery lung masks and train a U-Net-style model to predict a binary lung mask.

The detection and segmentation targets are lung anatomy, not tuberculosis lesions. The Montgomery TB label is retained as metadata for stratified splits; it is not the target for these two workflows.

## Environment

Environment files are stored in `environment/`:

- `environment/requirements.txt` for pip installs
- `environment/environment.yml` for the Conda environment used while developing the notebooks

## Datasets

The notebooks use two public chest-radiograph collections from the U.S. National Library of Medicine. In Google Colab, each notebook downloads and extracts the dataset(s) it needs automatically—open a notebook, select a GPU runtime, and use **Runtime → Run all**. No manual downloads, uploads, or path changes are needed. The first run downloads roughly 4.4 GB for classification (or the 0.6 GB Montgomery archive for detection and segmentation), so leave the setup cell running until it reports that extraction has started.

**Montgomery CXR dataset**  
Contains 138 posterior-anterior chest radiographs (80 normal and 58 with TB findings), image-level TB labels, and radiologist-supervised left- and right-lung masks. It supplies the anatomy annotations for segmentation and detection, and serves as the external cohort for classification.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

**Shenzhen CXR dataset**  
Contains 662 frontal chest radiographs (326 normal and 336 with TB findings) with image-level TB labels. It is used for classification training, validation, and held-out same-source testing.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

### Running Outside Colab

The same setup cell also works in a local Jupyter environment: it downloads data into `datasets/` beneath the directory from which the notebook is run. To use an existing local copy instead, change the dataset variables in that setup cell after it has run.

If you prefer a manual download, create one parent folder for the datasets and unzip the two archives there:

```bash
mkdir -p datasets
cd datasets

curl -LO https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
curl -LO https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

unzip ChinaSet_AllFiles.zip
unzip NLM-MontgomeryCXRSet.zip
```

After extraction, the folders should look like this:

```text
datasets/
+-- ChinaSet_AllFiles/
|   +-- CXR_png/
|   +-- ClinicalReadings/
+-- NLM-MontgomeryCXRSet/
    +-- MontgomerySet/
        +-- CXR_png/
        +-- ClinicalReadings/
        +-- ManualMask/
            +-- leftMask/
            +-- rightMask/
```

### Existing Local Paths

The default automatic setup needs no changes. The following paths are only useful if you deliberately want to reuse an existing local dataset copy.

If you downloaded the data into this repository under `datasets/`, use:

```python
SHENZHEN_DATASET_DIR = Path("datasets/ChinaSet_AllFiles")
MONTGOMERY_DATASET_DIR = Path("datasets/NLM-MontgomeryCXRSet/MontgomerySet")
```

For the detection and segmentation notebooks, only the Montgomery path is needed:

```python
DATASET_DIR = Path("datasets/NLM-MontgomeryCXRSet/MontgomerySet")
```

If you store the datasets somewhere else, keep the same final folder names and replace only the parent path. For example:

```python
SHENZHEN_DATASET_DIR = Path("/your/path/to/ChinaSet_AllFiles")
MONTGOMERY_DATASET_DIR = Path("/your/path/to/NLM-MontgomeryCXRSet/MontgomerySet")
```

## Notes

Colab storage is temporary. Downloaded data and trained model files are available for the current session and are removed when that runtime is reset unless you explicitly save them elsewhere.
