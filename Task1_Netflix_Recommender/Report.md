# Report: Netflix Content Recommendation System

**Auspify Technologies Internship, Task 1**

## 1. Problem and goal

Build a recommender that takes one Netflix title and suggests similar titles. The dataset has no user ratings or watch history, so the recommender is **content-based**: it only uses information about the titles themselves (genre, type, country, director, release decade and age rating). A recommender that uses what other people watched or rated would be called collaborative filtering, and it is not possible with this data.

The plan is to turn every title into a list of numbers (a vector), then measure how close two lists are. Close lists mean similar titles.

## 2. Dataset and cleaning

- File: `data/Dataset.csv`, with 8,790 rows and 10 columns: show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in
- `isnull().sum()` showed **zero** missing values. That is misleading. The file writes unknown values as the text "Not Given", and `isnull()` only counts empty cells. Counting the text directly shows **2,588 directors** and **287 countries** are "Not Given". I replaced "Not Given" with an empty string so it is not treated as a real director or country name
- I removed **3 repeated titles** (9-Feb, 15-Aug and 22-Jul, same title, type and release year), leaving **8,787 titles**. The same title twice would be a perfect match for itself and distort the results
- Each title has several genres stored in one string, like "Dramas, International Movies". Counting the whole strings splits the totals across many combinations ("Dramas, International Movies" appears 362 times, but "Dramas" alone appears 2,424 times after splitting). So I split `listed_in` before counting genres. The most common genres are International Movies (2,752), Dramas (2,424) and Comedies (1,673)

Screenshots: `screenshots/01_type_split.png`, `02_top_genres.png` and `03_ratings.png`.

## 3. Feature engineering

For every title I built one text string called the **soup**. It joins the genres, type, country, director, decade and rating into one line of words.

- **Genres are added twice.** Genre is the best sign that two titles are alike, so repeating it makes it count for more than the other features
- **Multi-word names use underscores** (for example `united_states` and `kirsten_johnson`). The underscore keeps a name as one feature. Without it, "United States" would split into "united" and "states", which would look like two unrelated words
- **The decade column** turns a release year like 2020 into `decade2020`, so titles from nearby years share a feature instead of having different year numbers

## 4. TF-IDF and similarity

**TF-IDF** gives each word in a soup a weight. A word gets a higher weight when it appears in this title but is rare across all titles. Words that appear everywhere tell us little, and rare words (like a director's name) are a stronger sign of similarity. I used scikit-learn's `TfidfVectorizer` with `token_pattern=r'\S+'`, so every underscore-joined name stays as one word.

The matrix has shape **(8787, 5140)**: 8,787 titles (rows) and 5,140 distinct words (columns) across all the soups.

For **Dick Johnson Is Dead**, the biggest weights are:

| Word            | Weight |
| --------------- | ------ |
| kirsten_johnson | 0.741  |
| documentaries   | 0.523  |
| pg-13           | 0.307  |
| decade2020      | 0.216  |
| united_states   | 0.158  |
| movie           | 0.107  |

The director's name has the highest weight because it is rare. "Movie" has the lowest because almost half of all titles share it.

**Cosine similarity** measures the angle between two vectors. A score close to 1 means very similar, and 0 means nothing in common. TF-IDF vectors are already scaled to length 1, so I used `linear_kernel`, which gives the same scores as cosine similarity and is faster.

**A title never recommends itself.** It shares every word with itself, so its score would be 1, the highest possible. The line `if j != idx` removes it from its own list.

## 5. Example recommendations

The `recommend(title, n=10)` function finds the title, scores it against every other title, sorts the scores from highest to lowest, and returns the top 10 as a table with the score.

- **Dick Johnson Is Dead:** all 10 recommendations are documentary movies, with scores from 0.577 (Murder to Mercy: The Cyntoia Brown Story) down to 0.518
- **Sankofa:** all 10 are Dramas and Independent Movies, and 9 of 10 are also International Movies, so the tags match well. But the scores are almost tied (0.501 down to 0.48), and the topics do not match. The model only sees tags like genre, rating, country and decade, and the dataset has no description column, so it cannot tell what a film is about
- **Paris Is Burning:** it is a documentary, but none of the recommendations were documentaries. Other tags (Classic Movies, Cult Movies) outweighed the single Documentaries tag

**Why some results have the same score:** titles with the same genres, type, rating, country and decade have identical soups, so they get identical scores. For example, Udaan and Garbage both scored 0.489. With stable sorting, tied titles stay in their original row order.

Screenshots: `screenshots/04_recommend_*.png` (8 titles).

## 6. Evaluation

I needed a number, not just a feeling that the suggestions look right. I used **rating precision@10**: for a chosen title, take its 10 recommendations and count how many have the **same age rating** as the chosen title.

Because rating is one of the soup features, using it would be cheating (the answer would be in the clues). So **the rating was hidden** from every version tested here.

- I tested 300 random titles (`seed=42`)
- **Random baseline:** if you picked 10 titles at random, the chance of the same rating is the sum of the squared share of each rating. This gives **0.219**

| Features used in the soup                 | Rating precision@10 |
| ----------------------------------------- | ------------------- |
| Genres only                               | 0.381               |
| Genres + type                             | 0.381               |
| Genres + type + country                   | 0.407               |
| All except rating (adds director, decade) | 0.388               |
| Random baseline                           | 0.219               |

Screenshots: `screenshots/05_evaluation_table.png` and `06_evaluation_chart.png`.

**What the results show:**

- Every version scores well above the random baseline (0.38 to 0.41 against 0.219). The recommendations are much more similar to the chosen title than random picks. The model never saw the rating, but still found titles with a matching rating more often than chance, so the features carry real signal
- **Genres + type + country did best (0.407).** Adding the country helped, probably because titles from the same country tend to have similar audience ratings
- Adding type to genres changed nothing (0.381). The genre names already contain words like "TV" and "Movies" (for example `tv_dramas`), so type is mostly repeated information
- Adding director and decade lowered the score a little (0.407 to 0.388). They are very specific features, so they may add noise for this particular test

**What this test does not show:** it does not prove the recommendations are good to watch, or that they match the topic of the title. Rating is only one thing titles can share, and the test partly measures audience level (a kids' show and a mature drama have different ratings).

## 7. Personalised suggestions

`recommend_for_user(liked_titles)` handles a person who likes several titles:

1. It finds the row for each liked title (missing titles are reported and skipped)
2. It **averages** those rows into one taste profile
3. It scores the profile against every title with cosine similarity
4. It removes the titles the user already likes and returns the top 10

Averaging works because each row shows which features a title has. The average row is partly like each liked title, so titles that share features with both end up closest to it.

- **User who likes Ganglands and Midnight Mass:** all 10 recommendations are TV shows rated TV-MA, and all include TV Horror, with most also tagged TV Dramas. These are the features the two titles have in common. Scores are 0.64 to 0.67
- **User who likes Dick Johnson Is Dead and Paris Is Burning:** all 10 recommendations are documentary movies. Scores are lower (0.54 to 0.59), probably because the two liked titles are quite different, so their average does not match any one title closely. The ratings are mixed (R, TV-MA, TV-14)

Screenshots: `screenshots/07_user_tv_fan.png` and `08_user_documentary_fan.png`.

## 8. Limitations and possible improvements

**Limitations**

- **No descriptions.** The model only sees tags, not what a title is about. Crip Camp, Reversing Roe and The Minimalists are all "documentaries", but their topics are very different
- **Many tied scores.** Titles with the same tags are impossible to tell apart
- **No real user data.** A "user" is just a list of titles I typed in. There is no watch history or time watched, so the profile shows what the titles have in common, not what a real person enjoys
- **The evaluation is rough.** Matching age ratings is only a stand-in for a good recommendation, and it was tested on 300 random titles

**Possible improvements**

- Add plot descriptions to the soup, so topics can be matched and ties can be broken
- Use real ratings or watch data to add collaborative filtering
- Test different weights, for example giving country or decade less or more influence

## 9. Files

`notebooks/recommender.ipynb` has all the code and outputs. `screenshots/` has 14 charts and tables. The environment is listed in `requirements.txt`.

GitHub repository: https://github.com/Hoyoshedah/netflix-recommender
