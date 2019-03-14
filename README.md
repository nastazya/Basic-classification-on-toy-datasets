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

