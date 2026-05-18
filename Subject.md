# Automatic Classification and Segmentation of Cancer Using Histopathology Images

## Context

Cancer causes millions of deaths each year worldwide. Accurate identification and timely treatment of cancer can save countless lives. However, achieving this remains challenging due to the complexity of diagnosis, the need for highly trained specialists, and the lengthy diagnostic process.

Automatic detection and classification of cancer types could help reduce these challenges, particularly in low-resource countries where there is a shortage of qualified medical personnel. Recent advances in artificial intelligence and deep learning have shown strong potential in analyzing histopathology images for cancer diagnosis and treatment planning.

## Aim

Develop a deep learning framework to segment and classify cells as malignant or healthy in the first dataset, while predicting survival outcomes (low vs. high probability) for the second dataset based on histopathological features.

## Requirements

In this project, two classification problems based on two histopathology datasets should be addressed.

The histopathology datasets can be downloaded using the following links:

- **BreaKHis**: https://nextcloud.univ-lille.fr/index.php/s/TXHR4qjo6PFamJC
- **DLBCL**: https://nextcloud.univ-lille.fr/index.php/s/CnGD56yX9WZtcAX

The CSV files of the train/test splits are available through the link below:
https://nextcloud.univ-lille.fr/index.php/s/dfdbEA3npkcB9oo

The first dataset is **BreaKHis**, a well-known breast cancer histopathology dataset containing microscopic images of breast tumor tissue. The objective will be to classify the samples as malignant (cancerous – Class 1) or benign (non-cancerous - Class 0).

The second dataset is **DLBCL**, which contains image patches of Diffuse Large B-Cell Lymphoma, a type of cancer affecting the immune system with significant global impact. The corresponding task will be a binary classification problem focused on predicting the survival outcome of patients (e.g., high (class 0) vs. low (class 1) survival rate).

To achieve these objectives, you may utilize transfer learning techniques by leveraging pre-trained models such as ResNet, EfficientNet, or Vision Transformers (ViT), which provide a solid foundation for medical image classification. For the segmentation task, architectures like U-Net, DeepLabv3+ or SAM are highly recommended to accurately isolate cellular structures. Additionally, you can implement data augmentation, such as rotations and color jittering, to improve model robustness.

## Tools

- **PyTorch** – Deep learning python library (https://pytorch.org/)
- **Draw diagrams**: https://app.diagrams.net/
- **Zotero** – Reference management software (https://www.zotero.org/)
- **JabRef** – Reference management software (https://www.jabref.org/)

## Planning and Deliverable

- **12h**: Introduction to the subject, including lectures and practical work
- **16h**: Design phase (preliminary studies and preparation)
  - Report (Word or LaTeX document) including the state-of-the-art section
  - **Be careful**: we expect your own writing (no copy-paste or use of generative AI text).
    - After 8 hours: First version of the "State of the Art" chapter
    - At 16 hours: Final version of the "State of the Art" chapter
  - The State-of-the-Art phase requires a comprehensive study of the dataset, alongside the identification and presentation of existing methodologies for classification and semantic segmentation. To ensure a broad technical exploration, each team member must select a distinct method, different from those chosen by their peers.
- **22h**: Implementation phase
- **8h**: Team integration, bug fixing, evaluation of results, and finalization of the report
- **2h**: Online presentation of each team's work

## Deliverable

1. **Report (~35 pages)**
   - Introduction
   - State of the Art
     - The aim is to present and explain all the methods and solutions studied in the articles that have been read. It is mandatory to cite your references correctly in the text. Be careful: we expect your own writing (no copy-paste or use of generative AI text).
     - A detailed comparative study, including the advantages and disadvantages of each presented method/solution (at least 2 pages).
   - Results obtained from each of the previous parts
   - Problems encountered
   - Conclusion

2. **PowerPoint of the Oral Presentation**

3. **Source code** of your implementation, along with graphics and the Word (.docx) or LaTeX (.tex) files

## Resources

**Google Scholar**: https://scholar.google.fr/

**Scientific repositories:**
- https://arxiv.org/
- https://www.sciencedirect.com/
- https://hal.archives-ouvertes.fr/
- http://citeseer.ist.psu.edu/index/
- https://ieeexplore.ieee.org/Xplore/home.jsp

## Useful References

- B. Guetarni et al., "A Vision Transformer-Based Framework for Knowledge Transfer From Multi-Modal to Mono-Modal Lymphoma Subtyping Models," in *IEEE Journal of Biomedical and Health Informatics*, vol. 28, no. 9, pp. 5562-5572, Sept. 2024, doi: 10.1109/JBHI.2024.3407878.

- Maximilian Springenberg, Annika Frommholz, Markus Wenzel, Eva Weicken, Jackie Ma, Nils Strodthoff, From modern CNNs to vision transformers: Assessing the performance, robustness, and classification strategies of deep learning models in histopathology, *Medical Image Analysis*, Volume 87, 2023, 102809, ISSN 1361-8415, https://doi.org/10.1016/j.media.2023.102809.

- Alfasly, S., Alabtah, G., Hemati, S. et al. Validation of histopathology foundation models through whole slide image retrieval. *Sci Rep* 15, 3990 (2025). https://doi.org/10.1038/s41598-025-88545-9

- Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L., A Dataset for Breast Cancer Histopathological Image Classification, *IEEE Transactions on Biomedical Engineering (TBME)*, 63(7):1455-1462, 2016.

- Fernandez-Pol, S., Natkunam, Y., Vrabac, D., Rojansky, R., Advani, R., Rajpurkar, P., S, & Ng, Andrew Y. (2022). H&E and immunohistochemical stain images of 209 cases of diffuse large B-cell lymphoma linked with cytogenetic features and clinical outcomes (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/NVA3-N783

- De Matos, Jonathan, et al. "Machine learning methods for histopathological image analysis: A review." *Electronics* 10.5 (2021): 562.
