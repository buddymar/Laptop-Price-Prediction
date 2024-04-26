# 💻 Cracking the Code: Predicting Laptop Prices with Analysis and Machine Learning Model
<br>

**Platform**: Jupyter Notebook | [Notebook via nbviewer](https://nbviewer.org/github/buddymar/Laptop-Price-Prediction/blob/main/Laptop%20Price%20Prediction.ipynb) | [Notebook via Github](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/Laptop%20Price%20Prediction.ipynb)<br>
**Programming Language**: Python <br>
**Libraries**: Pandas, NumPy, sklearn, Matplotlib, Seaborn, SHAP <br>
**Source Data**: Kaggle <br>
<br>

**Table of Contents**
- [Introduction](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-introduction)
- [Data Source](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-data-scraping)
- [Data Preprocessing](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-data-preprocessing)
	- [Data Cleaning](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-data-preprocessing)
      - [Missing Values](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#missing-values)
      - [Duplicated Values](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#duplicated-values)
	- [Data Transformation](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#correcting-errors--inconsistencies)
      - [Feature Engineering](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#correcting-errors--inconsistencies)
- [Exploratory Data Analysis](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-exploratory-data-analysis)
  - [Feature Target](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#player-attributes)
      - [Laptop Price]()
  - [Univariate & Bivariate Analysis](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#player-basic-stats)
      - [Company]()
      - [Laptop Type]()
      - [Display Size]()
      - [RAM]()
      - [Operation System]()
      - [Screen Resolution]()
      - [IPS Feature]()
      - [Touchscreen Feature]()
      - [CPU Brand]()
      - [CPU Speed]()
      - [GPU Brand]()
      - [Memory]()
      - [Weight]()
  - [Handling Outliers](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#player-advanced-stats)
      - [Laptop Price]()
      - [RAM 32 GB]()
      - [Resolution 2560x1440]()
  - [Features Correlations](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#team-stats)
- [Predictive Modeling](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#-predictive-modeling)
  - [Preprocessing](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#modeling)
  - [Modeling](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#modeling)
  - [Model Interpretation](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/README.md#model-interpretation)
<br>

---

## 📌 **Introduction**

<p align="center">
    <kbd> <img width="1000" alt="mvp banner" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/laptop_banner.jpg"> </kbd> <br>
</p>

Laptop telah menjadi kebutuhan primer di kalangan umum, didorong oleh perkembangan digital yang cepat dan dinamis. Saat membeli laptop, banyak komponen dan fitur yang memengaruhi harga totalnya, termasuk CPU, GPU, dan lainnya. Oleh karena itu, penting untuk memahami dampak setiap komponen ini pada harga laptop. Dengan informasi ini, kita dapat mengetahui komponen yang paling berpengaruh pada harga laptop, mengestimasi harga komponen tertentu, dan bahkan memperkirakan harga total laptop berdasarkan spesifikasinya.

Notebook ini bertujuan untuk menganalisis dan memodelkan faktor-faktor yang memengaruhi harga laptop di pasaran. Analisis ini akan mengidentifikasi fitur yang paling signifikan dalam menentukan harga laptop, serta membangun model untuk memprediksi harga berdasarkan fitur-fitur yang ada dalam dataset.

<br>

## 📌 **Data Source**

Berikut ini feature dan deskripsi dari dataset yang digunakan pada analisis ini.

Table 1 — Feature Engineering
 **Feature** | **Explanation** |
-----------------|--------------|
Company | Perusahaan produsen laptop
Product | Brand dan model
TypeName | Tipe laptop (notebook, gaming, dll.) 
Inches | Ukuran layar
ScreenResolution | Resolusi layar
Cpu | Central Processing Unit (CPU)
Ram | Ukuran RAM laptop 
Memory | Ukuran & tipe memori laptop (HDD, SSD, dll.) 
GPU | Graphics Processing Units (GPU)
OpSys | Sistem operasi
Weight | Berat laptop
Price_euros | Harga laptop (€)

<br>

## 📌 **Data Preprocessing**

The data cleaning section will involve various processes such as correcting errors, adjusting data types, handling missing values, managing duplicates, and so on. In the data transformation section, various processes will be performed, including adding new columns or features, handling outliers, encoding variables, correcting errors, and creating and transforming new columns.

<br>

### Missing Values

Jumlah baris dengan missing data pada dataset: 0

<br>

### Duplicated Values

Jumlah baris dengan data duplikat pada dataset: 0

<br>

### Feature Engineering

Pada bagian ini, akan dilakukan *feature engineering* terhadap beberapa fitur untuk mendapatkan dan menambah fitur-fitur laptop yang ada pada dataset ini.

Table 2 — Feature Engineering
 **Feature** | **New Engineered Feature** |
-----------------|--------------|
Screen Resolution | Resolution, IPS, Touchscreen
CPU | Cpu_GHz, Cpu_brand
Memory | Memory_1, SSD_1 (GB), HDD_1 (GB), Hybrid_1 (GB), Flash_1 (GB), Memory_2, Memory_2 (GB), Total_Memory (GB) 
GPU | Gpu_brand

<br>

## 📌 **Exploratory Data Analysis**

The Exploratory Data Analysis (EDA) will focus on analyzing all available data and information related to winning the NBA MVP award. It will primarily be divided into:
- Player Attributes
- Player Basic Stats
- Player Advanced Stats
- Teams Performance

<br>

### Player Attributes

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/vote_share.png"> </kbd> <br>
</p>

<p align="center">
    <kbd> <img width="1000" alt="pos" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/position.png"> </kbd> <br>
</p>

**Key Points:**
- MPV winners consistently receive high vote shares, often close to or exceeding 90%, indicating strong support from NBA voters for their MVP candidacy.
- Despite the subjective nature of MVP voting, the consistently high vote shares for winners suggest a certain level of consensus among voters regarding the most deserving candidate. However, there may still be biases or factors influencing the voting process, such as media coverage, team success, or individual narratives.
- While power forwards and point guards have historically dominated MVP awards from 2001 to 2023, the last three MVP winners have been centers. Centers are traditionally known for their defensive presence, rebounding, and rim protection, but recent MVP-winning centers also excel offensively, showcasing versatility in their skill sets.

<br>

### Player Basic Stats

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/basic%20stats.png"> </kbd> <br>
</p>

**Key Points:**
- MVP winners exhibit superior performance across various statistical categories compared to both MVP vote-getters and all players.
- **Statistical excellence, particularly in scoring, shooting efficiency, playmaking, and defensive contributions, appears to be a common trait among MVP winners.**

<br>

### Player Advanced Stats

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/adv%20stats.png"> </kbd> <br>
</p>

<p align="center">
    <kbd> <img width="1000" alt="pos" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/bpm.png"> </kbd> <br>
</p>

**Key Points:**
- Advanced/analytical statistics provide a more nuanced understanding of player performance, focusing on efficiency, usage, and impact on both ends of the court.
- MVP winners consistently demonstrate superior performance across various advanced metrics compared to both MVP vote-getters and all players, emphasizing their overall impact and contribution to their teams' success.
- It's undeniable that OBPM can provide more insight into a player's value than DBPM. A player can receive MVP votes solely based on their offensive performance, even if they have little defensive impact while on the court. However, it's worth noting that many MVP winners also have high DBPM, which sets them apart from other vote-getters.

<br>

### Team Stats

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/team%20stats.png"> </kbd> <br>
</p>

<p align="center">
    <kbd> <img width="1000" alt="pos" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/team%20win%20vs%20win%20share.png"> </kbd> <br>
</p>

**Key Points:**
- Teams with MVP-caliber players tend to outperform their counterparts in terms of offensive and defensive efficiency, margin of victory, win-loss records, and overall and conference standings.
- Higher overall and conference standings among MVP-winning teams reflect their ability to elevate team performance and competitiveness, positioning them as key leaders in guiding their teams to success within their respective conferences and the league as a whole.

<br>

## 📌 **Predictive Modeling**

To predict the NBA MVP from this dataset, three models will be compared: Random Forest, XGBoost, and Extra Trees Regressor. The best-performing model will then be used for predicting and tracking the current season's NBA MVP. Given the imbalanced nature of the target variable, which often includes many zero values, the Root Mean Squared Logarithmic Error (RMSLE) will be used as the scoring metric.

<br>

### Modeling

Table 2 — Model Scoring
 **Model** | **RMSLE** | **R-squared** | **Best Hyperparameters** |
-----------------|--------------|--------------|--------------|
RandomForestRegressor | 0.2333 | 0.9063 | 'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 100
ExtraTreesRegressor | 0.2408 | 0.9087 | 'max_depth': 7, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 6, 'n_estimators': 100
XGBRegressor | 0.155 | 0.9705 | 'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 63, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'subsample': 0.9
<br>

### Model Prediction

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/model%20prediction.png"> </kbd> <br>
</p>

**Key Points:**
- Based on the NBA MVP predictions for 2001-2023 winners, the XGBoost model had the best performance, achieving the highest accuracy with 22 correct predictions out of 23 winners. The Extra Trees model made 19 correct predictions, while the Random Forest model made 18 correct predictions.
- Interestingly, all three models incorrectly predicted the MVP for the 2017 season, selecting James Harden instead of Russell Westbrook.
- Moreover, all three models predict Nikola Jokić as the MVP for the current 2024 season (as of March), with a high predicted vote share.
- Overall, the models demonstrated high accuracy in predicting the NBA MVP, with XGBoost leading the pack with an accuracy ratio of 22 out of 23 (95.65%).

<br>

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/top3%20mvp.png"> </kbd> <br>
</p>

**Key Points (as of March 2024):**
- Nikola Jokić, with the highest PER, WS, and BPM, likely contributes significantly to the model predicting him as the number one of the top 10 MVP frontrunners.
- Shai Gilgeous-Alexander (2nd) and Giannis Antetokounmpo (3rd) still have a chance to move up in the MVP ranking ladder based on their actual basic stats, advanced stats, team standings, and vote share prediction.
- Luka Dončić leads in PTS per game (34.1) this season, with a considerable gap to the second highest, Shai Gilgeous-Alexander (30.9). This contributes to Dončić being in 4th place currently, despite his team's poor standings.

<br>

### Model Interpretation

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/shap%20value.png"> </kbd> <br>
</p>

The feature with the highest impact on the XGBoost model prediction is `adj_W%`, which represents the team's winning percentage adjusted by the total basic stats of the player. The impact gap from this feature to the next feature is considerable. Following closely are the next two impactful features for the model: `Total_AdvStat`, representing the total advanced stats for the player, and adjusted `BPM`, which indicates the Box Plus/Minus of the player.

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/shap%20bee.png"> </kbd> <br>
</p>

From the beeswarm plot, we can discern in greater detail the influence of each feature in this model. The plot vividly illustrates the wide range of impact from the `adj_W` feature. Additionally, it's apparent that features with high-value records exert the most influence on the model's predictions. Conversely, most low-value records result in zero impact across all features, except for `AST`, `Age`, and `Year`.

<br>

## 📌 **Dashboard**

Utilizing the same dataset analyzed earlier, this dashboard offers an additional perspective on NBA player performance and facilitates comparisons between player statistics. The dashboard consists of two sections: Leaderboard and Player Detailed Stats.

In the Leaderboard section, users can view the top 10 leaders in various statistics for each year or team. Meanwhile, the Player Stats section provides detailed information on selected player statistics, including the actual values, percentiles against other players each year, and comparisons to other players.

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/dash1.png"> </kbd> <br>
</p>

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/dash2.png"> </kbd> <br>
</p>
