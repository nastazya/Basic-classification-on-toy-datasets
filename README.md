# class8_homework
Loading multiple datasets from scikit-learn, visualizing them and performing PCA and three classification algorithms:
- KNeighbors
- GaussianNB
- SVC

## Instructions for running analyse.py using Docker
**1) Copy to local repository:**
- analyse.py 
- Dockerfile

**2) Run the script in this format:**
```
docker build -t <image_name> <path>
docker run -it -v /${PWD}:/${PWD} -w /${PWD} <image_name> <dataset name>
```
where  `<dataset name>` is optional. 
If no files were provided the script will analyse breast_cancer dataset

**3) Observe plots in these folders:**
- `hist` for histograms
- `scatter` for 2D scatter plots
- `corr`  for heatmap
- `box` for boxplots
- `3D` for 3D plots

## Pseudocode for KNeighbors algorithm

**input:**  k closest training examples in the feature space
       s
       s
**output:** class membership
**Statistical setting: ** Suppose we have pairs {\displaystyle (X_{1},Y_{1}),(X_{2},Y_{2}),\dots ,(X_{n},Y_{n})} {\displaystyle (X_{1},Y_{1}),(X_{2},Y_{2}),\dots ,(X_{n},Y_{n})} taking values in {\displaystyle \mathbb {R} ^{d}\times \{1,2\}} {\mathbb  {R}}^{d}\times \{1,2\}, where Y is the class label of X, so that {\displaystyle X|Y=r\sim P_{r}} X|Y=r\sim P_{r} for {\displaystyle r=1,2} r=1,2 (and probability distributions {\displaystyle P_{r}} P_{r}). Given some norm {\displaystyle \|\cdot \|} \|\cdot \| on {\displaystyle \mathbb {R} ^{d}} \mathbb {R} ^{d} and a point {\displaystyle x\in \mathbb {R} ^{d}} x\in {\mathbb  {R}}^{d}, let {\displaystyle (X_{(1)},Y_{(1)}),\dots ,(X_{(n)},Y_{(n)})} (X_{{(1)}},Y_{{(1)}}),\dots ,(X_{{(n)}},Y_{{(n)}}) be a reordering of the training data such that {\displaystyle \|X_{(1)}-x\|\leq \dots \leq \|X_{(n)}-x\|} \|X_{{(1)}}-x\|\leq \dots \leq \|X_{{(n)}}-x\|
- _k ← n_
