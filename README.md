### Notebooks - AI for Radiologic Image Tasks: Classification, Detection, and Segmentation

Datasets used:

Montgomery CXR dataset: contains 148 CXR images with lung masks and TB annotations (positive or negative)
- For the detection task the bounding boxes were derived from the segmentation masks
- For the classification task it is held out as external validation

Shenzhen CXR dataset: contains ~600 CXR images with TB annotations (positive or negative)
- For the classification task it is used for training and internal validation

Next steps:
- Add omnibin integration for metrics analysis

Montgomery Dataset
- Link: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

Shenzhen Dataset
- Link: https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

