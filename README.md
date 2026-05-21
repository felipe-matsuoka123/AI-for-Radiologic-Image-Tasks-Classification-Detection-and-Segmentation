# AI for Radiologic Image Tasks: Classification, Detection, and Segmentation


Educational notebooks for three common radiology AI workflows: image classification, object detection, and segmentation. The notebooks are designed to show the full path from dataset organization and preprocessing to model training, validation, and interpretation of results.

## Notebooks

| Task | Notebook | Open in Colab |
| --- | --- | --- |
| Classification | `classification.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/classification.ipynb) |
| Detection | `detection.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/detection.ipynb) |
| Segmentation | `segmentation.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/segmentation.ipynb) |

## Datasets

All notebooks use public chest X-ray datasets from the National Library of Medicine collection described in the same paper. The dataset files are not stored in this repository, so download them once and point the notebooks to the extracted folders.

**Montgomery CXR dataset**  
Contains chest X-ray images with lung masks and TB annotations. It is used for segmentation, detection, and as external validation for classification.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

**Shenzhen CXR dataset**  
Contains chest X-ray images with TB annotations. It is used for classification training and internal validation.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

### Download And Folder Setup

Create one parent folder for the datasets, download both ZIP files there, and unzip them:

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

### Notebook Paths

Each notebook has a setup cell near the top where you paste the dataset path for your machine.

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

When running in Colab, download the ZIP files into the Colab runtime or mount Google Drive, then update the same path variables before executing the workflow.
