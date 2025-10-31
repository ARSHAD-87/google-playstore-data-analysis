# Google Play Store Analysis Dashboard

**Live Project Link:** [**https://YourUsername.github.io/playstore-analytics-dashboard/**](https://YourUsername.github.io/playstore-analytics-dashboard/)

*(Note: You will get this link after completing Step 5)*

---

## 1. Project Overview

This project is an interactive Python dashboard analyzing the Google Play Store dataset. The goal is to uncover insights into app categories, user sentiment, and the key factors (like size, rating, and revenue) that drive app success.

The dashboard is built entirely in Python, using **Plotly** for interactive visualizations and **NLTK** for sentiment analysis. The final output is a single, responsive HTML file that is hosted on GitHub Pages.

### Screenshots

**Desktop View:**
![Desktop Dashboard View](screenshots/dashboard_desktop.png)

**Mobile View:**
![Mobile Dashboard View](screenshots/dashboard_mobile.png)

---

## 2. Datasets Used

* **`Play Store Data.csv`:** Contains app-level data (Category, Rating, Installs, Price, etc.) for over 10,000 apps.
* **`User Reviews.csv`:** Contains user review text, sentiment polarity, and subjectivity for apps.

---

## 3. Data Transformations & Feature Engineering

Before visualization, the data was heavily cleaned and processed:

* **Handling Missing Data:** Dropped rows with missing 'Rating' and filled other NaNs with the column's mode.
* **Type Conversion:**
    * **Installs:** Removed `+` and `,` characters and converted to numeric.
    * **Price:** Removed `$` and converted to numeric.
    * **Size:** Converted 'M' (megabytes) and 'k' (kilobytes) suffixes into a unified numeric 'MB' column.
* **Feature Engineering:**
    * **Revenue:** Created `Revenue` column (`Installs` * `Price`).
    * **Date Features:** Converted `Last Updated` to datetime and extracted `Year` and `Month` columns.
    * **Sentiment Score:** Used NLTK's VADER to calculate a compound `Sentiment_Score` for each user review.
* **Data Merging:** Merged the app data with the aggregated review data to link apps to their average sentiment scores.

---

## 4. KPIs & Visuals (Internship Task: Figs 11-16)

This project was extended with 6 advanced analytic visuals to meet internship requirements.

* **Fig 11: Rating vs. Reviews (Dual-Axis Bar)**
    * **KPI:** Popularity vs. Quality
    * **Insight:** High review counts don't always guarantee a perfect average rating, even for top-tier apps.

* **Fig 12: Global Install Penetration (Choropleth Map)**
    * **KPI:** Geographic Market Share
    * **Insight:** 'ENTERTAINMENT' app installs are highly concentrated in a few key countries, while 'EDUCATION' has a much wider global footprint.

* **Fig 13: Installs vs. Revenue (Dual-Axis Bar)**
    * **KPI:** Business Model Viability
    * **Insight:** 'GAME' revenue depends on high installs, but 'PRODUCTIVITY' apps can succeed with a high-price, niche-user model.

* **Fig 14: Install Growth (Time Series Line)**
    * **KPI:** Market Volatility
    * **Insight:** 'BUSINESS' app growth is volatile and event-driven (spiky), while 'ENTERTAINMENT' app growth is stable and predictable.

* **Fig 15: Size vs. Rating (Bubble Chart)**
    * **KPI:** User Tolerance
    * **Insight:** For popular apps, users clearly do not care about large file sizes as long as the quality (rating) is high.

* **Fig 16: Cumulative Installs (Stacked Area)**
    * **KPI:** Market Velocity
    * **Insight:** 'PHOTOGRAPHY' is the established market leader in installs, but 'PRODUCTIVITY' is the high-velocity challenger closing the gap.

---

## 5. How to Run This Project Locally

1.  Clone the repository:
    ```sh
    git clone [https://github.com/YourUsername/playstore-analytics-dashboard.git](https://github.com/YourUsername/playstore-analytics-dashboard.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd playstore-analytics-dashboard
    ```
3.  Install the required libraries:
    ```sh
    pip install pandas numpy plotly nltk
    ```
4.  Run the Python script:
    ```sh
    python google_play_analysis.py
    ```
5.  This will automatically generate and open the `index.html` file in your default web browser.
