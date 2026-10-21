"""
ml_models.py
------------
The six machine-learning models behind ShopSmart Analytics.

    1. Linear Regression -> predict future product sales
    2. Naive Bayes       -> classify customer reviews as positive / negative
    3. KNN               -> recommend products from similar customers / products
    4. SVM               -> classify customers likely to purchase or not
    5. Decision Tree     -> find customers who may respond to a promotion
    6. ANN (MLP)         -> predict purchase probability from behavior data
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, r2_score, mean_absolute_error, roc_auc_score)
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def clf_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, pos_label=1), 3),
        "recall": round(recall_score(y_true, y_pred, pos_label=1), 3),
        "f1": round(f1_score(y_true, y_pred, pos_label=1), 3),
    }
    if y_prob is not None:
        m["roc_auc"] = round(roc_auc_score(y_true, y_prob), 3)
    return m


# ---------------------------------------------------------------- 1. Linear Regression
class SalesForecaster:
    """One Linear Regression per product.
    Features: time index (trend), holiday flag, mid-year flag, ad spend."""

    FEATURES = ["t", "holiday", "midyear", "ad_spend"]

    def __init__(self, sales, products):
        self.products = products
        self.sales = sales.copy()
        self.sales["holiday"] = self.sales.month_num.isin([11, 12]).astype(int)
        self.sales["midyear"] = self.sales.month_num.isin([6, 7]).astype(int)
        self.models, self.r2, self.mae = {}, [], []
        for pid, g in self.sales.groupby("product_id"):
            # train on first 18 months, test on last 6 to report honest metrics
            train, test = g[g.t < 18], g[g.t >= 18]
            m = LinearRegression().fit(train[self.FEATURES], train.units_sold)
            pred = m.predict(test[self.FEATURES])
            self.r2.append(r2_score(test.units_sold, pred))
            self.mae.append(mean_absolute_error(test.units_sold, pred))
            # refit on all data for the real forecast
            self.models[pid] = LinearRegression().fit(g[self.FEATURES], g.units_sold)
        self.metrics = {"avg_r2": round(float(np.mean(self.r2)), 3),
                        "avg_mae_units": round(float(np.mean(self.mae)), 1)}

    def forecast(self, product_id, months_ahead=6, ad_spend=None):
        g = self.sales[self.sales.product_id == product_id].sort_values("t")
        last_t = int(g.t.max())
        last_month = pd.Period(g.month.iloc[-1], freq="M")
        spend = float(ad_spend if ad_spend is not None else g.ad_spend.tail(6).mean())
        future = []
        for k in range(1, months_ahead + 1):
            p = last_month + k
            future.append({"month": str(p), "t": last_t + k,
                           "holiday": int(p.month in (11, 12)),
                           "midyear": int(p.month in (6, 7)), "ad_spend": spend})
        fdf = pd.DataFrame(future)
        fdf["units_sold"] = np.maximum(self.models[product_id].predict(fdf[self.FEATURES]), 0).round()
        price = float(self.products.loc[self.products.product_id == product_id, "price"].iloc[0])
        return {
            "history": [{"month": r.month, "units": int(r.units_sold)} for r in g.itertuples()],
            "forecast": [{"month": r.month, "units": int(r.units_sold),
                          "revenue": round(r.units_sold * price, 2)} for r in fdf.itertuples()],
            "coefficients": dict(zip(self.FEATURES, np.round(self.models[product_id].coef_, 3).tolist())),
        }

    def total_forecast(self, months_ahead=6):
        """Company-wide revenue history + forecast (sum over products)."""
        hist = self.sales.groupby("month").revenue.sum().round(2)
        fut = {}
        for pid in self.models:
            for r in self.forecast(pid, months_ahead)["forecast"]:
                fut[r["month"]] = fut.get(r["month"], 0) + r["revenue"]
        return {"history": [{"month": m, "revenue": float(v)} for m, v in hist.items()],
                "forecast": [{"month": m, "revenue": round(v, 2)} for m, v in sorted(fut.items())]}


# ---------------------------------------------------------------- 2. Naive Bayes
class ReviewClassifier:
    """Bag-of-words (unigrams + bigrams) + Multinomial Naive Bayes."""

    def __init__(self, reviews):
        X_tr, X_te, y_tr, y_te = train_test_split(
            reviews.text, reviews.sentiment, test_size=0.25, random_state=1, stratify=reviews.sentiment)
        self.model = make_pipeline(CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                                   MultinomialNB(alpha=1.0))
        self.model.fit(X_tr, y_tr)
        pred = self.model.predict(X_te)
        self.metrics = clf_metrics((y_te == "positive").astype(int), (pred == "positive").astype(int))
        self.model.fit(reviews.text, reviews.sentiment)  # refit on everything
        self.reviews = reviews.assign(predicted=self.model.predict(reviews.text))

    def predict(self, text):
        probs = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        label = classes[int(np.argmax(probs))]
        return {"label": label, "confidence": round(float(np.max(probs)), 3),
                "probabilities": {c: round(float(p), 3) for c, p in zip(classes, probs)}}

    def top_words(self, n=8):
        vec, nb = self.model.named_steps["countvectorizer"], self.model.named_steps["multinomialnb"]
        words = np.array(vec.get_feature_names_out())
        classes = list(nb.classes_)
        diff = nb.feature_log_prob_[classes.index("positive")] - nb.feature_log_prob_[classes.index("negative")]
        order = np.argsort(diff)
        return {"positive": words[order[::-1][:n]].tolist(), "negative": words[order[:n]].tolist()}


# ---------------------------------------------------------------- 3. KNN
class Recommender:
    """User-based KNN (cosine) on the customer x product rating matrix, plus
    item-based KNN for 'customers who liked this also liked'."""

    def __init__(self, ratings, products, k=10):
        self.products = products.set_index("product_id")
        self.matrix = ratings.pivot_table(index="customer_id", columns="product_id",
                                          values="rating", fill_value=0)
        self.k = k
        self.user_knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(self.matrix.values)
        self.item_knn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(self.matrix.values.T)

    def customers(self):
        return self.matrix.index.tolist()

    def recommend(self, customer_id, n=5):
        if customer_id not in self.matrix.index:
            raise ValueError("customer has no ratings")
        row = self.matrix.loc[[customer_id]].values
        dist, idx = self.user_knn.kneighbors(row)
        neigh_idx, sims = idx[0][1:], 1 - dist[0][1:]  # drop the customer themself
        neigh = self.matrix.iloc[neigh_idx]
        # similarity-weighted average rating from neighbors
        weighted = (neigh.values * sims[:, None]).sum(0) / (sims.sum() + 1e-9)
        scores = pd.Series(weighted, index=self.matrix.columns)
        already = self.matrix.loc[customer_id] > 0
        scores = scores[~already].sort_values(ascending=False).head(n)
        rated = self.matrix.loc[customer_id]
        return {
            "rated": [self._p(pid, rating=int(r)) for pid, r in rated[rated > 0].items()],
            "recommendations": [self._p(pid, score=round(float(s), 2)) for pid, s in scores.items()],
            "neighbors": [{"customer_id": int(self.matrix.index[i]), "similarity": round(float(s), 3)}
                          for i, s in zip(neigh_idx[:5], sims[:5])],
        }

    def similar_products(self, product_id):
        col = list(self.matrix.columns).index(product_id)
        dist, idx = self.item_knn.kneighbors(self.matrix.values.T[[col]])
        return [self._p(int(self.matrix.columns[i]), similarity=round(float(1 - d), 3))
                for i, d in zip(idx[0][1:], dist[0][1:])]

    def _p(self, pid, **extra):
        p = self.products.loc[int(pid)]
        return {"product_id": int(pid), "name": p["name"], "category": p.category,
                "price": float(p.price), **extra}


# ---------------------------------------------------------------- 4. SVM
SVM_FEATURES = ["age", "income", "tenure_months", "past_purchases",
                "sessions_per_month", "cart_adds", "days_since_last_visit"]


class PurchaseSVM:
    """RBF-kernel SVM: will this customer purchase (1) or not (0)?"""

    def __init__(self, customers):
        X, y = customers[SVM_FEATURES], customers.will_purchase
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
        self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        self.model.fit(X_tr, y_tr)
        self.metrics = clf_metrics(y_te, self.model.predict(X_te))
        self.model.fit(X, y)

    def predict(self, features):
        x = pd.DataFrame([features])[SVM_FEATURES]
        label = int(self.model.predict(x)[0])
        margin = float(self.model.decision_function(x)[0])
        return {"will_purchase": label, "label": "Likely to purchase" if label else "Not likely to purchase",
                "margin": round(margin, 3)}


# ---------------------------------------------------------------- 5. Decision Tree
TREE_FEATURES = ["age", "income", "email_open_rate", "used_coupon_before",
                 "days_since_last_visit", "past_purchases", "tenure_months"]


class PromoTree:
    """Shallow decision tree so the rules stay readable for marketing."""

    def __init__(self, customers):
        X, y = customers[TREE_FEATURES], customers.responded_to_promo
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=3, stratify=y)
        self.model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0)
        self.model.fit(X_tr, y_tr)
        self.metrics = clf_metrics(y_te, self.model.predict(X_te))
        self.model.fit(X, y)

    def importances(self):
        imp = sorted(zip(TREE_FEATURES, self.model.feature_importances_), key=lambda t: -t[1])
        return [{"feature": f, "importance": round(float(v), 3)} for f, v in imp if v > 0]

    def rules(self):
        return export_text(self.model, feature_names=TREE_FEATURES, decimals=2)

    def predict(self, features):
        x = pd.DataFrame([features])[TREE_FEATURES]
        prob = float(self.model.predict_proba(x)[0][1])
        return {"will_respond": int(prob >= 0.5), "probability": round(prob, 3),
                "path": self._path(x)}

    def _path(self, x):
        tree, node_ids = self.model.tree_, self.model.decision_path(x).indices
        steps = []
        for node in node_ids:
            if tree.children_left[node] == tree.children_right[node]:  # leaf
                break
            f, thr = TREE_FEATURES[tree.feature[node]], tree.threshold[node]
            val = float(x.iloc[0][f])
            steps.append(f"{f} = {val:g} {'<=' if val <= thr else '>'} {thr:.2f}")
        return steps


# ---------------------------------------------------------------- 6. ANN
ANN_FEATURES = ["sessions_per_month", "avg_session_min", "pages_per_session", "cart_adds",
                "wishlist_items", "days_since_last_visit", "past_purchases", "email_open_rate"]


class PurchaseANN:
    """Multi-layer perceptron (16 -> 8 hidden, ReLU) giving a purchase probability."""

    def __init__(self, customers):
        X, y = customers[ANN_FEATURES], customers.will_purchase
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", alpha=1e-3,
                          max_iter=1500, early_stopping=True, random_state=0))
        self.model.fit(X_tr, y_tr)
        prob = self.model.predict_proba(X_te)[:, 1]
        self.metrics = clf_metrics(y_te, (prob >= 0.5).astype(int), prob)
        self.model.fit(X, y)
        self.loss_curve = [round(float(v), 4) for v in self.model.named_steps["mlpclassifier"].loss_curve_]

    def predict_proba(self, df):
        return self.model.predict_proba(df[ANN_FEATURES])[:, 1]

    def predict(self, features):
        p = float(self.predict_proba(pd.DataFrame([features]))[0])
        return {"probability": round(p, 3), "label": "High intent" if p >= 0.7
                else "Medium intent" if p >= 0.4 else "Low intent"}


# ---------------------------------------------------------------- train everything
def train_all():
    products, sales = load("products"), load("sales")
    reviews, customers, ratings = load("reviews"), load("customers"), load("ratings")
    return {
        "data": {"products": products, "sales": sales, "reviews": reviews,
                 "customers": customers, "ratings": ratings},
        "lr": SalesForecaster(sales, products),
        "nb": ReviewClassifier(reviews),
        "knn": Recommender(ratings, products),
        "svm": PurchaseSVM(customers),
        "tree": PromoTree(customers),
        "ann": PurchaseANN(customers),
    }


if __name__ == "__main__":
    m = train_all()
    for key, name in [("lr", "Linear Regression"), ("nb", "Naive Bayes"), ("svm", "SVM"),
                      ("tree", "Decision Tree"), ("ann", "ANN")]:
        print(f"{name:18s} {m[key].metrics}")
    print("KNN example:", m["knn"].recommend(1)["recommendations"][:3])
