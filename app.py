"""
Fake Social Media Profile Detector
Backend: ML model + Flask REST API
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from flask import send_from_directory
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────

def generate_dataset(n_samples=5000, random_state=42):
    """
    Generate a realistic synthetic dataset of social media profiles.
    Features are engineered to reflect real-world indicators of fake accounts.
    """
    np.random.seed(random_state)
    n_real = n_samples // 2
    n_fake = n_samples - n_real

    def real_profiles(n):
        return {
            # Account basics
            "followers_count":        np.random.lognormal(6.0, 1.5, n).astype(int),
            "following_count":        np.random.lognormal(5.5, 1.2, n).astype(int),
            "post_count":             np.random.lognormal(4.5, 1.0, n).astype(int),
            "account_age_days":       np.random.randint(180, 3650, n),

            # Profile completeness
            "has_profile_pic":        np.random.choice([0, 1], n, p=[0.05, 0.95]),
            "bio_length":             np.random.randint(30, 250, n),
            "has_website_link":       np.random.choice([0, 1], n, p=[0.55, 0.45]),
            "is_verified":            np.random.choice([0, 1], n, p=[0.92, 0.08]),

            # Activity patterns
            "avg_posts_per_week":     np.random.uniform(0.5, 15.0, n),
            "avg_likes_per_post":     np.random.lognormal(3.0, 1.2, n),
            "avg_comments_per_post":  np.random.lognormal(1.5, 1.0, n),
            "avg_shares_per_post":    np.random.lognormal(1.0, 1.0, n),

            # Engagement & authenticity
            "engagement_rate":        np.random.uniform(0.01, 0.12, n),
            "reply_to_comments_rate": np.random.uniform(0.2, 0.9, n),
            "pct_original_posts":     np.random.uniform(0.4, 1.0, n),
            "posting_hour_variance":  np.random.uniform(4.0, 10.0, n),

            # Network signals
            "mutual_friends_ratio":   np.random.uniform(0.05, 0.6, n),
            "spam_report_count":      np.random.poisson(0.3, n),
            "login_location_changes": np.random.poisson(2, n),

            # Username & name
            "username_digit_ratio":   np.random.uniform(0.0, 0.15, n),
            "username_length":        np.random.randint(5, 20, n),
            "name_matches_username":  np.random.choice([0, 1], n, p=[0.3, 0.7]),

            # Label
            "label": np.zeros(n, dtype=int)
        }

    def fake_profiles(n):
        return {
            "followers_count":        np.random.lognormal(3.5, 2.0, n).astype(int),
            "following_count":        np.random.lognormal(6.5, 1.0, n).astype(int),
            "post_count":             np.random.randint(0, 30, n),
            "account_age_days":       np.random.randint(1, 180, n),

            "has_profile_pic":        np.random.choice([0, 1], n, p=[0.45, 0.55]),
            "bio_length":             np.random.choice([0, np.random.randint(5, 40)], n),
            "has_website_link":       np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "is_verified":            np.zeros(n, dtype=int),

            "avg_posts_per_week":     np.random.uniform(0.0, 50.0, n),
            "avg_likes_per_post":     np.random.lognormal(0.5, 1.5, n),
            "avg_comments_per_post":  np.random.lognormal(0.2, 0.8, n),
            "avg_shares_per_post":    np.random.lognormal(0.1, 0.5, n),

            "engagement_rate":        np.random.uniform(0.0, 0.02, n),
            "reply_to_comments_rate": np.random.uniform(0.0, 0.1, n),
            "pct_original_posts":     np.random.uniform(0.0, 0.3, n),
            "posting_hour_variance":  np.random.uniform(0.1, 2.0, n),

            "mutual_friends_ratio":   np.random.uniform(0.0, 0.05, n),
            "spam_report_count":      np.random.poisson(3, n),
            "login_location_changes": np.random.poisson(8, n),

            "username_digit_ratio":   np.random.uniform(0.2, 0.8, n),
            "username_length":        np.random.randint(8, 30, n),
            "name_matches_username":  np.random.choice([0, 1], n, p=[0.8, 0.2]),

            "label": np.ones(n, dtype=int)
        }

    real = pd.DataFrame(real_profiles(n_real))
    fake = pd.DataFrame(fake_profiles(n_fake))

    # Fix bio_length for fake (it's a scalar issue)
    fake["bio_length"] = np.where(
        np.random.random(n_fake) < 0.6,
        0,
        np.random.randint(5, 50, n_fake)
    )

    df = pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=random_state)

    # Derived features
    df["follower_following_ratio"] = df["followers_count"] / (df["following_count"] + 1)
    df["posts_per_day"] = df["post_count"] / (df["account_age_days"] + 1)
    df["like_to_follower_ratio"] = df["avg_likes_per_post"] / (df["followers_count"] + 1)
    df["suspicious_activity_score"] = (
        df["spam_report_count"] * 2 +
        df["login_location_changes"] * 0.5 +
        df["username_digit_ratio"] * 3
    )

    return df


FEATURE_COLS = [
    "followers_count", "following_count", "post_count", "account_age_days",
    "has_profile_pic", "bio_length", "has_website_link", "is_verified",
    "avg_posts_per_week", "avg_likes_per_post", "avg_comments_per_post",
    "avg_shares_per_post", "engagement_rate", "reply_to_comments_rate",
    "pct_original_posts", "posting_hour_variance", "mutual_friends_ratio",
    "spam_report_count", "login_location_changes", "username_digit_ratio",
    "username_length", "name_matches_username",
    "follower_following_ratio", "posts_per_day",
    "like_to_follower_ratio", "suspicious_activity_score"
]


# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────

def train_model():
    print("🔄 Generating dataset...")
    df = generate_dataset(n_samples=10000)

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 Training Random Forest + Gradient Boosting ensemble...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    gb_acc = accuracy_score(y_test, gb.predict(X_test))

    print(f"  RF Accuracy:  {rf_acc:.4f}")
    print(f"  GB Accuracy:  {gb_acc:.4f}")
    print(classification_report(y_test, rf.predict(X_test), target_names=["Real", "Fake"]))

    # Save models
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(gb, "gb_model.pkl")
    joblib.dump(FEATURE_COLS, "feature_cols.pkl")

    # Save feature importance
    importance = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
    with open("feature_importance.json", "w") as f:
        json.dump(importance, f)

    print("✅ Models saved.")
    return rf, gb


# ─────────────────────────────────────────
# LOAD / TRAIN
# ─────────────────────────────────────────

if os.path.exists("rf_model.pkl") and os.path.exists("gb_model.pkl"):
    print("📦 Loading saved models...")
    rf_model = joblib.load("rf_model.pkl")
    gb_model = joblib.load("gb_model.pkl")
    with open("feature_importance.json") as f:
        feature_importance = json.load(f)
else:
    rf_model, gb_model = train_model()
    with open("feature_importance.json") as f:
        feature_importance = json.load(f)


# ─────────────────────────────────────────
# HELPER: engineer features from raw input
# ─────────────────────────────────────────

def engineer_features(data: dict) -> pd.DataFrame:
    row = {col: data.get(col, 0) for col in FEATURE_COLS[:22]}  # raw cols
    followers = float(row.get("followers_count", 0))
    following = float(row.get("following_count", 1))
    posts = float(row.get("post_count", 0))
    age = float(row.get("account_age_days", 1))
    likes = float(row.get("avg_likes_per_post", 0))
    spam = float(row.get("spam_report_count", 0))
    loc = float(row.get("login_location_changes", 0))
    digit_r = float(row.get("username_digit_ratio", 0))

    row["follower_following_ratio"] = followers / (following + 1)
    row["posts_per_day"] = posts / (age + 1)
    row["like_to_follower_ratio"] = likes / (followers + 1)
    row["suspicious_activity_score"] = spam * 2 + loc * 0.5 + digit_r * 3

    return pd.DataFrame([row])[FEATURE_COLS]


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = engineer_features(data)

        rf_proba = rf_model.predict_proba(df)[0]
        gb_proba = gb_model.predict_proba(df)[0]

        # Ensemble: weighted average (RF 60%, GB 40%)
        fake_prob = 0.6 * rf_proba[1] + 0.4 * gb_proba[1]
        real_prob = 1.0 - fake_prob

        label = "Fake" if fake_prob > 0.5 else "Real"
        trust_score = int(round(real_prob * 100))
        confidence = float(max(fake_prob, real_prob))

        # Risk factors
        risk_flags = []
        row = data
        if float(row.get("account_age_days", 999)) < 30:
            risk_flags.append("Account is very new (< 30 days)")
        if float(row.get("followers_count", 0)) < 10 and float(row.get("following_count", 0)) > 500:
            risk_flags.append("Extreme following/follower imbalance")
        if float(row.get("spam_report_count", 0)) > 3:
            risk_flags.append("Multiple spam reports")
        if float(row.get("engagement_rate", 1)) < 0.005:
            risk_flags.append("Very low engagement rate")
        if float(row.get("username_digit_ratio", 0)) > 0.4:
            risk_flags.append("Username contains many digits (bot-like)")
        if int(row.get("has_profile_pic", 1)) == 0:
            risk_flags.append("No profile picture")
        if float(row.get("bio_length", 99)) < 5:
            risk_flags.append("No or very short bio")
        if float(row.get("login_location_changes", 0)) > 10:
            risk_flags.append("Suspicious login location changes")
        if float(row.get("pct_original_posts", 1)) < 0.1:
            risk_flags.append("Mostly non-original content (reposts/spam)")

        return jsonify({
            "label": label,
            "trust_score": trust_score,
            "fake_probability": round(fake_prob * 100, 1),
            "real_probability": round(real_prob * 100, 1),
            "confidence": round(confidence * 100, 1),
            "risk_flags": risk_flags,
            "feature_importance": feature_importance
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForest + GradientBoosting Ensemble"})


@app.route("/api/sample", methods=["GET"])
def sample():
    """Return sample profiles for demo"""
    profiles = {
        "real": {
            "followers_count": 1420, "following_count": 380, "post_count": 245,
            "account_age_days": 1100, "has_profile_pic": 1, "bio_length": 95,
            "has_website_link": 1, "is_verified": 0, "avg_posts_per_week": 3.2,
            "avg_likes_per_post": 87, "avg_comments_per_post": 12,
            "avg_shares_per_post": 5, "engagement_rate": 0.065,
            "reply_to_comments_rate": 0.6, "pct_original_posts": 0.85,
            "posting_hour_variance": 7.2, "mutual_friends_ratio": 0.22,
            "spam_report_count": 0, "login_location_changes": 2,
            "username_digit_ratio": 0.05, "username_length": 12,
            "name_matches_username": 1
        },
        "fake": {
            "followers_count": 14, "following_count": 4980, "post_count": 3,
            "account_age_days": 12, "has_profile_pic": 0, "bio_length": 0,
            "has_website_link": 0, "is_verified": 0, "avg_posts_per_week": 0.2,
            "avg_likes_per_post": 1, "avg_comments_per_post": 0,
            "avg_shares_per_post": 0, "engagement_rate": 0.001,
            "reply_to_comments_rate": 0.0, "pct_original_posts": 0.0,
            "posting_hour_variance": 0.3, "mutual_friends_ratio": 0.0,
            "spam_report_count": 5, "login_location_changes": 15,
            "username_digit_ratio": 0.62, "username_length": 22,
            "name_matches_username": 0
        }
    }
    return jsonify(profiles)


if __name__ == "__main__":
    print("\n🚀 Starting Fake Profile Detector API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
