# ShopSmart Analytics — Online Shopping ML Dashboard

An e-commerce analytics web app that **predicts customer behavior, recommends products,
and helps businesses manage inventory**, using six machine-learning algorithms.

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.
The first start generates the sample data (`data/*.csv`) and trains all six models (a few seconds).

## Algorithms used

| # | Algorithm | Task in this project | Where in the dashboard |
|---|-----------|----------------------|------------------------|
| 1 | **Linear Regression** | Predict future product sales (next 6 months) from trend, holiday season and ad spend | *Sales forecast* — total revenue chart, per-product forecast, ad-spend slider, stock tip |
| 2 | **Naive Bayes** (Multinomial) | Classify customer reviews as positive or negative | *Review classification* — sentiment by category, "classify a review" box, top words |
| 3 | **KNN** (cosine) | Recommend products based on similar customers (user-based) and similar products (item-based) | *Product recommendations* |
| 4 | **SVM** (RBF kernel) | Classify customers likely to purchase or not purchase | *Customer insights* — SVM form, "Likely buyers" KPI |
| 5 | **Decision Tree** (depth 4) | Determine which customers may respond to a promotion | *Customer insights* — tree form with decision path, feature importance, rules |
| 6 | **ANN** (MLP, 16→8 hidden) | Predict customer purchase probability from behavior data | *Customer insights* — probability form, distribution chart, top-intent table |

Every classifier is evaluated on a held-out 25% test set; the sales model is tested on the
last 6 months. Scores appear in the **Model performance** table at the bottom of the page.

## Project files

```
app.py            Flask web server + JSON API
ml_models.py      The six models (train, evaluate, predict)
generate_data.py  Creates the sample datasets in data/
templates/        index.html — dashboard page
static/           style.css, app.js (charts use Chart.js)
data/             products, sales, reviews, customers, ratings (CSV)
```

Run `python ml_models.py` to train the models and print their scores in the terminal.

## Sample results

| Model | Result |
|-------|--------|
| Linear Regression | R² ≈ 0.53, mean error ≈ 20 units/month |
| Naive Bayes | accuracy ≈ 0.97 |
| SVM | accuracy ≈ 0.82 |
| Decision Tree | accuracy ≈ 0.87 |
| ANN | accuracy ≈ 0.78, ROC-AUC ≈ 0.87 |

*All data is synthetic, generated for demonstration.*
