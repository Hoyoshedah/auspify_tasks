# Netflix Content Recommendation System

Auspify Technologies Internship, Task 1 (Easy).
A content-based recommender that suggests Netflix titles similar to a chosen title, using genres, type, country, director, release decade and rating.

## Dataset

Dataset.csv: 8,790 Netflix titles with 10 columns (type, title, director, country, release_year, rating, duration, listed_in, and so on). After cleaning: 8,787 titles.

## How it works

1. Clean the data (replace "Not Given" with empty text, remove 3 duplicate titles)
2. Build a text "soup" for each title (genres, type, country, director, decade, rating)
3. Convert the soups to numbers with TF-IDF
4. Score titles with cosine similarity and return the top 10
5. Evaluate with a held-out feature (rating) against a random baseline
6. Personalised suggestions: average several liked titles into a taste profile

## How to run

1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. Open notebooks/recommender.ipynb and choose Run All

## Project structure

- data/ the dataset
- notebooks/ the notebook with all code and notes
- screenshots/ charts, recommendation tables, evaluation results
- report.md project report
- requirements.txt libraries used

## Results (rating precision@10, rating hidden from the model)

Random baseline 0.219; genres only 0.381; genres + type 0.381; genres + type + country 0.407; all except rating 0.388.

## Limitations

The dataset has no description or cast columns, so the model matches tags, not what a title is about. Many titles tie in score. There is no user data, so personalisation is simulated from a list of liked titles.

## Repository

https://github.com/Hoyoshedah/netflix-recommender
