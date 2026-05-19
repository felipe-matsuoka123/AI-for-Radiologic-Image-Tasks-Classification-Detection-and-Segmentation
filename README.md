# AI for Radiologic Image Tasks: Classification, Detection, and Segmentation


Educational notebooks for three common radiology AI workflows: image classification, object detection, and segmentation. The notebooks are designed to show the full path from dataset organization and preprocessing to model training, validation, and interpretation of results.

## Notebooks

| Task | Notebook | Open in Colab |
| --- | --- | --- |
| Classification | `classification.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/classification.ipynb) |
| Detection | `detection.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/detection.ipynb) |
| Segmentation | `segmentation.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/felipe-matsuoka123/AI-for-Radiologic-Image-Tasks-Classification-Detection-and-Segmentation/blob/main/segmentation.ipynb) |

## Datasets

**Montgomery CXR dataset**  
Contains chest X-ray images with lung masks and TB annotations. It is used for segmentation, detection, and as external validation for classification.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

**Shenzhen CXR dataset**  
Contains chest X-ray images with TB annotations. It is used for classification training and internal validation.

- Dataset: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

## Notes

The notebooks use local dataset paths by default. When running in Colab, update the path setup cells or mount/download the datasets before executing the workflow.
