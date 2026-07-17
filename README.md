# AI for Radiologic Image Tasks: Classification, Detection, and Segmentation


Companion notebooks for the manuscript, *AI for Radiologic Image Tasks: Classification, Detection, and Segmentation*. The manuscript explains the concepts, annotation requirements, evaluation principles, and limitations of each task. These notebooks provide the hands-on counterpart: reproducible, small-scale examples that take the reader from the public datasets to preprocessing, model training, held-out evaluation, and qualitative review.

They are educational workflows, not clinical decision-support systems or deployment templates. In particular, the small public datasets and simplified splits are useful for learning, but do not replace patient-level evaluation, external validation, calibration, regulatory review, or clinical workflow integration.

## Notebooks

| Task | Notebook | Open in Colab |
| :--- | :--- | :---: |
| Classification | [`classification.ipynb`](notebooks/classification.ipynb) | [![Open classification in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/classification.ipynb) |
| Detection | [`detection_yolo.ipynb`](notebooks/detection_yolo.ipynb) | [![Open detection in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/detection_yolo.ipynb) |
| Segmentation | [`segmentation.ipynb`](notebooks/segmentation.ipynb) | [![Open segmentation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/notebooks/segmentation.ipynb) |

- **Classification:** train on Shenzhen TB labels and compare the held-out Shenzhen test set with the independent Montgomery cohort.
- **Detection:** convert Montgomery left- and right-lung masks to boxes, then fine-tune a pretrained YOLO detector for the single `lung` class.
- **Segmentation:** combine the Montgomery lung masks and train a U-Net-style model to predict a binary lung mask.

The detection and segmentation targets are lung anatomy, not tuberculosis lesions. The Montgomery TB label is retained as metadata for stratified splits; it is not the target for these two workflows.

## Environment

Environment files are stored in `environment/`:

- `environment/requirements.txt` for pip installs
- `environment/environment.yml` for the Conda environment used while developing the notebooks

## Datasets

The notebooks use two public chest-radiograph collections from the U.S. National Library of Medicine. Colab-ready ZIP archives are publicly hosted on [Hugging Face](https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr). In Google Colab, open a notebook, select a GPU runtime, and use **Runtime → Run all**—no manual downloads, uploads, accounts, or path changes are needed. The first run downloads only the archive required for that workflow.

**Montgomery CXR dataset**  
Contains 138 posterior-anterior chest radiographs (80 normal and 58 with TB findings), image-level TB labels, and radiologist-supervised left- and right-lung masks. It supplies the anatomy annotations for segmentation and detection, and serves as the external cohort for classification.

- Colab-ready public copy: https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/tree/main/MontgomerySet
- Direct archive: https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/resolve/main/MontgomerySet.zip?download=true
- Original NLM source: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

**Shenzhen CXR dataset**  
Contains 662 frontal chest radiographs (326 normal and 336 with TB findings) with image-level TB labels. It is used for classification training, validation, and held-out same-source testing.

- Colab-ready public copy: https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/tree/main/ChinaSet_AllFiles
- Direct archive: https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/resolve/main/ChinaSet_AllFiles.zip?download=true
- Original NLM source: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

### Running Outside Colab

The same setup cell works in local Jupyter: it downloads public files into `datasets/` beneath the directory from which the notebook runs. The folder layout is:

```text
datasets/
+-- ChinaSet_AllFiles/
|   +-- CXR_png/
|   +-- ClinicalReadings/
+-- MontgomerySet/
    +-- CXR_png/
    +-- ClinicalReadings/
    +-- ManualMask/
        +-- leftMask/
        +-- rightMask/
```

## Notes

Colab storage is temporary. Downloaded data and trained model files are available for the current session and are removed when that runtime is reset unless you explicitly save them elsewhere.
