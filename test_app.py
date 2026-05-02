import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'

def test_sample_endpoint(client):
    response = client.get('/api/sample')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'real' in data
    assert 'fake' in data

def test_predict_real_profile(client):
    payload = {
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
    }
    response = client.post('/api/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'label' in data
    assert 'trust_score' in data
    assert data['label'] == 'Real'

def test_predict_fake_profile(client):
    payload = {
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
    response = client.post('/api/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['label'] == 'Fake'

def test_predict_missing_fields(client):
    response = client.post('/api/predict',
                           data=json.dumps({}),
                           content_type='application/json')
    assert response.status_code == 200  # app handles missing fields with defaults