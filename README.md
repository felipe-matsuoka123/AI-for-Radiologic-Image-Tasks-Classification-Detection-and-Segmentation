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

## Beginner Quick Start

No programming experience is required to follow the main workflow. Each notebook is designed to run from top to bottom in Google Colab.

1. Open one of the notebooks using its **Open in Colab** button.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**.
4. Read the explanation above each result before changing any settings.

The notebooks use the following visual cues:

- **🔒 Setup or implementation:** run the collapsed cell without editing it. You may expand it if you want to study the code.
- **✏️ Control panel:** safe settings for beginner experiments. Changing image size, batch size, or epoch count can alter memory use and runtime.
- **👀 How to read this result:** a plain-language guide to the table, figure, or numbers produced by the preceding cell.
- **🧪 Try it yourself:** an optional experiment and the effect you should expect.
- **⚠️ Watch for:** a common mistake or limitation.

## The Three Tasks at a Glance

```text
Classification: chest X-ray → one TB probability
Detection:      chest X-ray → lung bounding boxes + confidence scores
Segmentation:   chest X-ray → one lung/background label for every pixel
```

| Task | Training annotation | Model output | Example question |
| :--- | :--- | :--- | :--- |
| Classification | One label for the entire image | One probability | “Does this image contain findings consistent with TB?” |
| Detection | One rectangle around each object | Boxes, classes, and confidence scores | “Where are the lung fields?” |
| Segmentation | A label for every pixel | A pixel-level mask | “Which pixels belong to the lungs?” |

All three notebooks follow the same broad pipeline:

```text
Images + annotations → Prepare data → Train → Validate → Test → Inspect errors
```

Training data updates the model. Validation data guides model selection and settings. Test data is held aside until the end so it can provide a less biased final evaluation.

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

## Glossary

**Annotation:** Information attached to an image that provides the correct answer during supervised learning. It may be an image-level label, a bounding box, or a pixel mask.

**Augmentation:** A random, realistic modification of a training image, such as a small rotation or brightness change. It creates variety without collecting new images. Spatial annotations must undergo the same geometric change as the image.

**Batch:** A small group of images processed together during one model update. A larger batch can use more GPU memory.

**Checkpoint:** A saved copy of the model from a particular stage of training. These notebooks retain the checkpoint that performs best on validation data.

**Confidence score:** The detector's estimate of how strongly a predicted box contains the target class. It is not the same as a clinically calibrated probability.

**Confidence threshold:** The minimum confidence score required to keep a detection. Raising it usually removes more uncertain boxes but can also miss real objects.

**Dice score:** A segmentation overlap score comparing a predicted mask with the reference mask. A value of 1 means perfect overlap and 0 means no overlap.

**Epoch:** One complete pass through the training images.

**Intersection over Union (IoU):** The area shared by a prediction and its reference divided by their combined area. Higher values indicate better spatial overlap. IoU is used for both boxes and masks.

**Learning rate:** The size of each model update during training. A value that is too large can make training unstable; one that is too small can make learning very slow.

**Logit:** A model's raw numerical output before it is converted to a probability. A sigmoid function converts the binary logits in these notebooks to values between 0 and 1.

**Loss:** A number representing prediction error during optimization. Training attempts to reduce it. A lower loss is generally better, but the held-out metrics and qualitative results must also be examined.

**Mean average precision (mAP):** A detection summary that combines precision and recall across confidence levels. `mAP50` uses an IoU matching threshold of 0.50; `mAP50-95` averages results over stricter IoU thresholds and is therefore harder.

**Pretrained model:** A model that has already learned general image features from another dataset and is then adapted to the current task. This process is called transfer learning.

**Tensor:** A multidimensional array used by a deep-learning framework. An image batch commonly has the shape `[batch, channels, height, width]`.

**Training, validation, and test sets:** Non-overlapping data groups with different roles. Training updates model weights, validation supports model selection, and testing estimates final performance after model selection.

**Overfitting:** A model learning the training data too specifically. One warning sign is improving training performance alongside worsening validation performance.

## Troubleshooting

### Training is very slow

Confirm that Colab shows a GPU under **Runtime → Change runtime type**. The setup output should report a CUDA device. CPU execution is supported but model training may take substantially longer.

### The runtime runs out of memory

Reduce `BATCH_SIZE` first. If necessary, also reduce `IMAGE_SIZE` or use a smaller model preset. Restart the runtime after an out-of-memory error because GPU memory may remain occupied.

### A dataset or pretrained model fails to download

Check the runtime's internet connection and rerun the failed cell. Temporary errors from Colab, Hugging Face, or the model host usually do not require changing notebook code. Files already downloaded during the current session are reused.

### An import fails after packages were installed

Run the two setup cells in order. If Colab reports that a newly installed package requires a restart, select **Runtime → Restart session**, then use **Run all** again.

### A file or directory is not found outside Colab

Start Jupyter from the repository root, or confirm that the automatically downloaded `datasets/` directory is beneath the notebook's current working directory. Avoid moving individual notebooks away from the project unless you also update the data path.

### A data-loading cell appears frozen

Set `NUM_WORKERS = 0` in the relevant control panel. This is slower but avoids multiprocessing issues in some local Jupyter environments.

### Results differ between runs

The notebooks set random seeds, but GPU operations and library versions can still introduce small differences. Focus on the overall pattern rather than expecting every decimal value or sampled image to be identical.

### Training stopped when the Colab runtime disconnected

Colab storage is temporary. A disconnected or reset runtime may lose downloaded data and saved checkpoints. Rerun the notebook or copy important result files to persistent storage before ending the session.

### Training metrics improve but validation metrics worsen

This pattern can indicate overfitting. Use the validation-selected checkpoint rather than the final epoch, and consider fewer epochs, stronger augmentation, or a smaller model. Do not use the test set to choose these settings.

### Detection produces no boxes or too many boxes

Inspect the confidence threshold. Lowering it keeps more predictions and may recover missed lungs; raising it removes low-confidence boxes. Changing this threshold affects the displayed predictions but does not retrain the detector.

### A metric looks high but a prediction looks wrong

Summary metrics average across cases and can hide important failure modes. Always inspect the qualitative examples. These small educational datasets are not sufficient to establish clinical performance.

## Notes

Colab storage is temporary. Downloaded data and trained model files are available for the current session and are removed when that runtime is reset unless you explicitly save them elsewhere.
