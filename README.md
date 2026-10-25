# Semantic Segmentation of Aerial Imagery

**A deep learning project utilizing a U-Net architecture, and a U-Net extended with CBAM, for 2D semantic segmentation of aerial imagery over the city of Potsdam, Germany.**

The aim of this project is to build, train and evaluate two semantic segmentation models using the *ISPRS* Potsdam dataset, consisting of very high resolution (VHR) True Ortho Photos (TOP). The two models are built and trained from scratch. 


Models to build, train and evaluate:
- A classical Vanilla U-Net approach, inspired by the paper *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Extending the U-Net by introducing Convolutional Block Attention Modules (CBAM) to the skip-connections. Inspired by the paper *CBAM: Convolutional Block Attention Module*.


Segmentation of VHR images is particularly challenging as small objects and small-scale surface texture become visible, leading to high within-class variability and at the same time low inter-class differences. Therefore, it will be of interest to investigate how well a Vanilla U-Net handles the challenge of semantic segmentation of the VHR Potsdam images, as well as to investigate whether extending the U-Net with CBAM on skip-connections will produce better results or not. 
