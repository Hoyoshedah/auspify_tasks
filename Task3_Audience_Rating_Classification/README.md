# Netflix Audience Rating Classification

**Auspify Technologies Internship, Task 3**

A machine learning project that predicts the **audience group** of a Netflix title (Kids, Family, Teens or Adults) from its details, using a Decision Tree and a Random Forest. Each model is tested in a default version and a tuned version.

## What this project does

1. Groups the 14 rating codes (G, PG, TV-MA and so on) into 4 audience groups
2. Builds 49 features from the dataset columns
3. Trains a Decision Tree and a Random Forest
4. Tunes both with GridSearchCV and 5-fold cross-validation
5. Compares all models against a baseline and studies the mistakes

## Dataset

- File: `data/Dataset.csv`
- 8,790 rows, 10 columns. After removing 3 repeated titles: 8,787 rows
- 82 titles rated NR or UR have no clear audience group, so they were removed. That leaves **8,705 titles**

| Audience group | Ratings in it                  | Titles |
| -------------- | ------------------------------ | ------ |
| Kids           | G, TV-Y, TV-Y7, TV-Y7-FV, TV-G | 906    |
| Family         | PG, TV-PG                      | 1,148  |
| Teens          | PG-13, TV-14                   | 2,645  |
| Adults         | R, NC-17, TV-MA                | 4,006  |

The groups are unbalanced: Adults is 46.0% of the data and Kids is only 10.4%. So I used a stratified split and macro F1, not accuracy alone.

## Important note: the rating is the target

`rating` is what the model predicts, so it is **not** used as a feature. Using it would give the answer away (data leakage).

## Results (20% test set, 1,741 titles)

| Model                    | Accuracy | Macro F1 |
| ------------------------ | -------- | -------- |
| Baseline (always Adults) | 0.460    | 0.158    |
| Decision Tree (default)  | 0.566    | 0.540    |
| Random Forest (default)  | 0.620    | 0.586    |
| Decision Tree (tuned)    | 0.619    | 0.592    |
| Random Forest (tuned)    | 0.635    | 0.586    |

- Every model beats the baseline
- Tuning helped the Decision Tree most (accuracy 0.566 to 0.619)
- The tuned Random Forest has the best accuracy (0.635). The tuned Decision Tree has a slightly higher macro F1 (0.592)
- Kids and Adults are predicted best. Family and Teens are the hardest, because they overlap with the groups next to them

## Folder structure

```
Audience Rating Classifier/
  data/Dataset.csv
  notebooks/rating_classifier.ipynb
  screenshots/        (8 charts and tables)
  README.md
  Report.md
  requirements.txt
  .gitignore
```

## How to run it (Windows PowerShell)

```powershell
cd "D:\Audience Rating Classifier"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks\rating_classifier.ipynb
```

Then click **Run All**.

## Tools

Python, pandas, numpy, scikit-learn, matplotlib, Jupyter, VS Code, Git/GitHub

## Limitations

- Family and Teens overlap with their neighbours, so even a good model finds them hard
- The dataset has no description or cast column, which could help a lot
- The grouping of ratings into 4 audiences is my own choice. A different grouping would change the results

## Links

GitHub repo: https://github.com/Hoyoshedah/Audience-Rating-Classifier
