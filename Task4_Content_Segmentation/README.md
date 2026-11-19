# Task 4: Netflix Content Segmentation

Group Netflix titles into meaningful clusters using unsupervised machine learning (K-Means).

## Project structure
- `data/Dataset.csv`: Netflix titles dataset (8,790 rows, 10 columns)
- `notebooks/content_segmentation.ipynb`: full analysis, all outputs saved
- `screenshots/`: elbow and silhouette plots, cluster sizes, PCA plot, cluster profile
- `Report.md`: method, results and interpretation
- `requirements.txt`: Python packages used

## How to run
1. Install the packages: `pip install -r requirements.txt`
2. Open `notebooks/content_segmentation.ipynb` and run all cells.

## Workflow
1. Prepare numerical features (release year, movie minutes, TV seasons) and categorical features (type, rating, main genre, main country), one-hot encoded.
2. Scale with StandardScaler and apply K-Means, choosing k with the elbow method and silhouette score.
3. Identify the content groups (10 clusters).
4. Visualize the clusters in 2D with PCA.
5. Interpret each cluster by its average numbers and most common categories.

## Result in one line
k = 10 gave the best silhouette score (0.217) and produced clear groups such as mature comedy movies, family movies, crime TV shows and two kinds of kids' TV.


## GitHub
https://github.com/Hoyoshedah/Netflix-Content-Segmentation
