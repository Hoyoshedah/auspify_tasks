# Report: Audience Rating Classification

**Auspify Technologies Internship, Task 3**

## 1. Goal

Build machine learning models that predict the audience group of a Netflix title (**Kids, Family, Teens or Adults**) from its other details. I used a Decision Tree and a Random Forest, and compared a default version of each with a tuned version.

## 2. The data

- 8,790 rows and 10 columns: show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in
- I removed 3 repeated titles (same title, type and release year), leaving 8,787 rows
- The text "Not Given" was turned into real blanks
- The dataset has 14 different rating codes. Four audience groups are easier to understand, so I grouped them first:

| Group  | Ratings in it                  | Titles | Share |
| ------ | ------------------------------ | ------ | ----- |
| Kids   | G, TV-Y, TV-Y7, TV-Y7-FV, TV-G | 906    | 10.4% |
| Family | PG, TV-PG                      | 1,148  | 13.2% |
| Teens  | PG-13, TV-14                   | 2,645  | 30.4% |
| Adults | R, NC-17, TV-MA                | 4,006  | 46.0% |

- The ratings **NR** and **UR** (not rated and unrated) do not show which group a title belongs to. I removed those 82 titles, which leaves **8,705 titles**
- The groups are **unbalanced**. Adults is almost half of the data and Kids is about one tenth. So accuracy alone can mislead (see section 6)

Screenshots: `screenshots/01_rating_counts.png` and `screenshots/02_audience_groups.png`.

## 3. Step 1: Choose the features

`rating` is the **target** (what the model predicts), so it is not used as a feature. If it were, the model could read the answer. This is called **data leakage**.

The model uses **49 features**:

| Feature                 | How it was made                                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| is_tv_show              | 1 for TV Show, 0 for Movie                                                                                  |
| release_year            | Used as it is                                                                                               |
| year_added, month_added | Taken from `date_added`. Blanks were filled with the median                                                 |
| has_director            | 1 if a director is listed, 0 if not                                                                         |
| duration_num            | The number from "90 min" or "2 Seasons". Fine to use here, because the target is the audience, not the type |
| country (11 columns)    | First country, top 10 plus "Other", one-hot encoded                                                         |
| genres (32 columns)     | Cleaned from `listed_in` and turned into 0/1 columns with CountVectorizer                                   |

For the genres I removed the words "tv", "movies" and "shows" and joined multi-word names with an underscore (for example `children_family`).

## 4. Step 2: Split the data

I split the data into 80% training (6,964 titles) and 20% test (1,741 titles). I used `stratify` so both parts keep the same group shares (46.0% Adults, 30.4% Teens, 13.2% Family, 10.4% Kids), and `random_state=42` so the split can be repeated.

## 5. Step 3: Train and tune the models

- **Baseline:** always predict "Adults"
- **Decision Tree (default):** a chain of yes/no questions, with no limits
- **Random Forest (default):** 200 trees that vote
- **Tuned versions:** I used **GridSearchCV** with 5-fold cross-validation (shuffled and stratified). It tries every combination of settings and keeps the best one. I scored with **macro F1**, because the groups are unbalanced

Settings I tried:

| Model         | Settings tried                                                                         |
| ------------- | -------------------------------------------------------------------------------------- |
| Decision Tree | max_depth: 3, 5, 8, 12, None. min_samples_leaf: 1, 5, 10, 20. criterion: gini, entropy |
| Random Forest | n_estimators: 100, 200. max_depth: 10, None. min_samples_leaf: 1, 3, 5                 |

Best settings (`screenshots/07_best_settings.png`):

| Setting          | Decision Tree   | Random Forest   |
| ---------------- | --------------- | --------------- |
| criterion        | entropy         | -               |
| max_depth        | None (no limit) | None (no limit) |
| min_samples_leaf | 20              | 3               |
| n_estimators     | -               | 100             |
| Best CV macro F1 | 0.596           | 0.614           |

The cross-validation scores are measured on the training data, so they are a little different from the final test scores below.

## 6. Step 4: Results on the test set

| Model                    | Accuracy | Macro F1 |
| ------------------------ | -------- | -------- |
| Baseline (always Adults) | 0.460    | 0.158    |
| Decision Tree (default)  | 0.566    | 0.540    |
| Random Forest (default)  | 0.620    | 0.586    |
| Decision Tree (tuned)    | 0.619    | 0.592    |
| Random Forest (tuned)    | 0.635    | 0.586    |

Screenshots: `screenshots/03_model_comparison_table.png` and `screenshots/04_model_comparison.png`.

- **Why I used macro F1 as well as accuracy:** the baseline gets 0.460 accuracy, which looks fine, but its macro F1 is only 0.158, because it never finds a Kids, Family or Teens title. Macro F1 gives every group the same weight, so it shows this
- Every model beats the baseline
- **Tuning helped the Decision Tree a lot** (accuracy 0.566 to 0.619, macro F1 0.540 to 0.592). Limiting the tree stopped it from memorising the training data (overfitting)
- **Tuning helped the Random Forest only a little** (accuracy 0.620 to 0.635, macro F1 stayed at 0.586)
- The tuned Random Forest has the best accuracy. The tuned Decision Tree has a slightly higher macro F1. The two are close, and the tree is easier to explain

## 7. Step 5: Where the mistakes are

Tuned Random Forest, per group (`screenshots/08_classification_report.png`):

| Group  | Precision | Recall | F1    | Titles in test set |
| ------ | --------- | ------ | ----- | ------------------ |
| Kids   | 0.830     | 0.702  | 0.760 | 181                |
| Family | 0.579     | 0.270  | 0.368 | 230                |
| Teens  | 0.552     | 0.420  | 0.477 | 529                |
| Adults | 0.643     | 0.866  | 0.738 | 801                |

Confusion matrix (`screenshots/05_confusion_matrix.png`). Rows are the true group and columns are what the model predicted:

| True group | Kids | Family | Teens | Adults |
| ---------- | ---- | ------ | ----- | ------ |
| Kids       | 127  | 22     | 8     | 24     |
| Family     | 25   | 62     | 68    | 75     |
| Teens      | 1    | 20     | 222   | 286    |
| Adults     | 0    | 3      | 104   | 694    |

- **Kids and Adults are found best.** They are the two ends of the scale, so they look the most different. Only 8 real Kids titles were called Teens
- **Family is the weakest** (recall 0.270). Of 230 Family titles, only 62 were found. 68 were called Teens and 75 were called Adults
- **Many Teens titles are called Adults** (286 of 529). When the model is unsure, it leans towards the biggest group
- The high recall for Adults (0.866) is partly because the model guesses "Adults" so often, which also lowers the precision for Adults (0.643)

## 8. What the model looks at

The feature chart for the tuned Random Forest (`screenshots/06_feature_importance.png`) shows that the most important features are `duration_num`, the kids genre columns (`genre_kids`, `genre_children_family`), `release_year`, `month_added` and `year_added`. This makes sense: kids' titles tend to be shorter and have kids' genres. Feature importance shows what the model **used**, not what **causes** a rating.

## 9. Limitations

- Family and Teens overlap with their neighbouring groups, so they are hard to predict, and the model is much better at the two ends (Kids and Adults)
- The groups are unbalanced. I did not use class weights, which could help the small groups
- There is no description or cast column in the dataset, which could help a lot
- My grouping of ratings into 4 audiences is a choice. A different grouping would change the results
- With 1,741 test titles, small differences between models could be noise

## 10. What I learned

- **Grouping the target:** turning 14 rating codes into 4 groups made the problem easier to understand
- **Class imbalance:** a model that always guesses the biggest group still gets 46% accuracy. Macro F1 shows that it is not learning
- **Stratified split:** it keeps the same group shares in the training and test sets
- **Overfitting and tuning:** a Decision Tree with no limits memorises the training data. GridSearchCV with cross-validation helped me pick settings that work better on new data
- **Reading mistakes:** the confusion matrix showed that the model confuses neighbouring groups (Family and Teens), not groups that are far apart

## 11. Files

`notebooks/rating_classifier.ipynb` has all the code. `screenshots/` has 8 charts and tables. The environment is listed in `requirements.txt`.

GitHub repository: https://github.com/Hoyoshedah/Audience-Rating-Classifier
