# AI for Radiologic Image Tasks: Classification, Detection, and Segmentation


Companion notebooks for the manuscript, *AI for Radiologic Image Tasks: Classification, Detection, and Segmentation*. The manuscript explains the concepts, annotation requirements, evaluation principles, and limitations of each task. These notebooks provide the hands-on counterpart: reproducible, small-scale examples that take the reader from the public datasets to preprocessing, model training, held-out evaluation, and qualitative review.

They are educational workflows, not clinical decision-support systems or deployment templates. In particular, the small public datasets and simplified splits are useful for learning, but do not replace patient-level evaluation, external validation, calibration, regulatory review, or clinical workflow integration.

## Notebooks

| Manuscript task | Notebook | What it demonstrates | Open in Colab |
| --- | --- | --- |
| Classification | `notebooks/classification.ipynb` | Train on Shenzhen TB labels; compare a held-out Shenzhen test set with the independent Montgomery cohort. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/classification.ipynb) |
| Detection | `notebooks/detection_yolo.ipynb` | Convert Montgomery left- and right-lung masks to boxes, then fine-tune a pretrained YOLO detector for the single `lung` class. This is the detection workflow described in the manuscript. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/detection_yolo.ipynb) |
| Segmentation | `notebooks/segmentation.ipynb` | Combine the Montgomery lung masks and train a U-Net-style model to predict a binary lung mask. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/segmentation.ipynb) |
| Supplemental detection exercise | `notebooks/detection.ipynb` | A fixed two-box ResNet-34 coordinate-regression exercise. It is retained as an introductory alternative and is not the YOLO workflow reported in the manuscript. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/detection.ipynb) |

The detection and segmentation targets are lung anatomy, not tuberculosis lesions. The Montgomery TB label is retained as metadata for stratified splits; it is not the target for these two workflows.

## Environment

Environment files are stored in `environment/`:

- `environment/requirements.txt` for pip installs
- `environment/environment.yml` for the Conda environment used while developing the notebooks

## Datasets

The notebooks use the two public chest-radiograph collections from the U.S. National Library of Medicine described in the manuscript. Dataset files are not stored in this repository; download them once and point the notebooks to the extracted folders.

**Montgomery CXR dataset**  
Contains 138 posterior-anterior chest radiographs (80 normal and 58 with TB findings), image-level TB labels, and radiologist-supervised left- and right-lung masks. It supplies the anatomy annotations for segmentation and detection, and serves as the external cohort for classification.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

**Shenzhen CXR dataset**  
Contains 662 frontal chest radiographs (326 normal and 336 with TB findings) with image-level TB labels. It is used for classification training, validation, and held-out same-source testing.

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
