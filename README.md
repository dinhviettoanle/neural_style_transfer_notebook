# Neural style transfer
_Sciences des Données et de la Décision 2021/2022
Algorithms in Machine Learning Notebook_

> **This notebook is quite compute intensive: it might be better to run it on Google Colab (or on Kaggle, depending on your Colab recent usage).** <br>
> The notebook is normally self-sufficient (all data is downloaded in the first cell and images are online), so you can upload only the `.ipynb` on Colab (or Kaggle).
> After running the first cell, you can check that the directory tree is similar to the one indicated below.

## Data

If you want to run it locally (bad idea if CUDA is not compatible), all the data is provided in this repository, except the images from the COCO Dataset. To download them, run `utils.py`.

## Directory tree

You may find this directory tree in your Colab session:

    .
    ├── images/
    │   ├── content-images/     # Content images
    │   ├── gif/                # Gif files (for section 2)
    │   └── style-images/       # Style images
    ├── models/                 # Empty directory that will store models (for section 2)
    ├── results/                # Some results
    ├── saved_models/           # Pre-trained models (for section 2)
    ├── solutions/              # Code solutions for the notebook
    ├── train/
    │   └── coco/               # 1000 images from the COCO dataset (for section 2)
    └── utils.py


