# Cell_Seg_Count-CMU




##MoNuSeg Dataset
MoNuSeg is a carefully annotated tissue image dataset of several patients with tumors of different organs and who were diagnosed at multiple hospitals. This dataset was created by downloading H&E stained tissue images captured at 40x magnification from TCGA archive. H&E staining is a routine protocol to enhance the contrast of a tissue section and is commonly used for tumor assessment (grading, staging, etc.). Given the diversity of nuclei appearances across multiple organs and patients, and the richness of staining protocols adopted at multiple hospitals, the training datatset will enable the development of robust and generalizable nuclei segmentation techniques that will work right out of the box.
https://monuseg.grand-challenge.org/Data/
1) Images (30) - Type *tif
2) Binary Masks (30)- Type *xml
   Total Annotations: 22,000

##CryoNuSeg Dataset
CryoNuSeg is the first fully annotated dataset of frozen H&E-Stained histological images. The dataset includes 30 image patches with a fixed size of 512x512 pixels of 10 human organs.
https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
1) Tissue Image (3) - Type *tif
2) Annotater 1 Annotations - Mask Binary - Type *png
  Total Annotations: 30,000

Example Task 
