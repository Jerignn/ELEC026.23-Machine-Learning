"""
app.py
------
ShopSmart Analytics - Flask web app serving the shopping analytics dashboard.

    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000
"""

import os
import numpy as np
from flask import Flask, jsonify, render_template, request

import generate_data
from ml_models import train_all, SVM_FEATURES, TREE_FEATURES, ANN_FEATURES

if not os.path.exists(os.path.join(generate_data.DATA_DIR, "customers.csv")):
    generate_data.main()

print("Training models...")
M = train_all()
D = M["data"]
print("Models ready.")

app = Flask(__name__)


def customer_scores():
    """Run every customer through SVM, Decision Tree and ANN."""
    c = D["customers"].copy()
    c["svm_purchase"] = M["svm"].model.predict(c[SVM_FEATURES])
    c["promo_prob"] = M["tree"].model.predict_proba(c[TREE_FEATURES])[:, 1]
    c["ann_prob"] = M["ann"].predict_proba(c)
    return c


SCORES = customer_scores()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def dashboard():
    sales, products, reviews = D["sales"], D["products"], M["nb"].reviews
    total = M["lr"].total_forecast(6)
    last_month = sales.month.max()

    # review sentiment by category (Naive Bayes predictions)
    rv = reviews.merge(products[["product_id", "category"]], on="product_id")
    by_cat = (rv.groupby("category").predicted.value_counts().unstack(fill_value=0)
              .reindex(columns=["positive", "negative"], fill_value=0))

    # ANN probability histogram (10 buckets)
    hist, edges = np.histogram(SCORES.ann_prob, bins=10, range=(0, 1))

    top = SCORES.sort_values("ann_prob", ascending=False).head(12)
    return jsonify({
        "kpis": {
            "revenue_last_month": round(float(sales[sales.month == last_month].revenue.sum()), 2),
            "revenue_next_month": total["forecast"][0]["revenue"],
            "positive_review_pct": round(float((reviews.predicted == "positive").mean() * 100), 1),
            "likely_buyers": int(SCORES.svm_purchase.sum()),
            "promo_responders": int((SCORES.promo_prob >= 0.5).sum()),
            "avg_purchase_prob": round(float(SCORES.ann_prob.mean() * 100), 1),
            "customers": int(len(SCORES)),
        },
        "metrics": {
            "Linear Regression": M["lr"].metrics,
            "Naive Bayes": M["nb"].metrics,
            "SVM": M["svm"].metrics,
            "Decision Tree": M["tree"].metrics,
            "ANN": M["ann"].metrics,
        },
        "revenue": total,
        "products": products.to_dict("records"),
        "reviews": {
            "by_category": [{"category": k, "positive": int(r.positive), "negative": int(r.negative)}
                            for k, r in by_cat.iterrows()],
            "top_words": M["nb"].top_words(),
            "recent": reviews.tail(8)[["text", "sentiment", "predicted"]].iloc[::-1].to_dict("records"),
        },
        "knn_customers": M["knn"].customers()[:300],
        "tree": {"importances": M["tree"].importances(), "rules": M["tree"].rules()},
        "ann": {
            "histogram": [{"bucket": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "count": int(h)}
                          for i, h in enumerate(hist)],
            "loss_curve": M["ann"].loss_curve,
        },
        "top_customers": [{
            "customer_id": int(r.customer_id), "age": int(r.age),
            "favorite_category": r.favorite_category,
            "ann_prob": round(float(r.ann_prob), 3),
            "svm_purchase": int(r.svm_purchase),
            "promo_prob": round(float(r.promo_prob), 3),
        } for r in top.itertuples()],
    })


@app.route("/api/forecast/<int:pid>")
def forecast(pid):
    spend = request.args.get("ad_spend", type=float)
    return jsonify(M["lr"].forecast(pid, 6, spend))


@app.route("/api/review", methods=["POST"])
def classify_review():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Please type a review."}), 400
    return jsonify(M["nb"].predict(text))


@app.route("/api/recommend/<int:cid>")
def recommend(cid):
    try:
        return jsonify(M["knn"].recommend(cid))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/similar/<int:pid>")
def similar(pid):
    return jsonify(M["knn"].similar_products(pid))


def numeric_payload(keys):
    body = request.get_json(silent=True) or {}
    return {k: float(body.get(k, 0)) for k in keys}


@app.route("/api/predict/svm", methods=["POST"])
def predict_svm():
    return jsonify(M["svm"].predict(numeric_payload(SVM_FEATURES)))


@app.route("/api/predict/tree", methods=["POST"])
def predict_tree():
    return jsonify(M["tree"].predict(numeric_payload(TREE_FEATURES)))


@app.route("/api/predict/ann", methods=["POST"])
def predict_ann():
    return jsonify(M["ann"].predict(numeric_payload(ANN_FEATURES)))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
