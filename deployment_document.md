# ProfileGuard — Deployment Document

## 1. Application Overview
ProfileGuard is a machine learning-powered web application that detects fake social media accounts. It analyses up to 22 profile attributes such as follower/following ratio, account age, engagement rate, spam reports, and username patterns to determine whether a social media profile is real or fake.
The application is useful for social media platforms, researchers, and individuals who want to verify the authenticity of online accounts. It returns a Trust Score (0–100), a verdict (Real or Fake), confidence percentage, and a list of risk flags explaining why the account was flagged.

### API Endpoints
| Method | URL | Description | Example Response |
|--------|-----|-------------|-----------------|
| GET | / | Serves the frontend HTML page | HTML page |
| GET | /api/health | Health check — confirms app is running | `{"status": "ok", "model": "RandomForest + GradientBoosting Ensemble"}` |
| POST | /api/predict | Accepts 22 profile attributes and returns ML prediction | `{"label": "Fake", "trust_score": 12, "fake_probability": 88.0, "risk_flags": [...]}` |
| GET | /api/sample | Returns two sample profiles (real and fake) for demo | `{"real": {...}, "fake": {...}}` |

## 2. Architecture Diagram
```
User Browser
     |
     | HTTP Request (port 5000)
     v
AWS EC2 Instance (t2.micro, Ubuntu 22.04)
IP: 16.171.253.13
     |
     | Docker (--restart=always)
     v
Docker Container (profileguard:v1)
     |
     | Python Flask (0.0.0.0:5000)
     v
Flask Application (app.py)
     |
     |-- GET  /             --> frontend/index.html
     |-- GET  /api/health   --> JSON status
     |-- POST /api/predict  --> ML Ensemble Model
     |-- GET  /api/sample   --> Sample profiles
     |
     v
ML Models (Random Forest 60% + Gradient Boosting 40%)
Trained on 10,000 synthetic social media profiles
```

## 3. Tools and Technologies
| Tool | Version | Purpose |
|------|---------|---------|
| Linux (Ubuntu) | 22.04 LTS | Operating system for both development and EC2 server |
| Python | 3.11 | Backend programming language for Flask and ML pipeline |
| Flask | 3.1 | Lightweight web framework used to build the REST API |
| Flask-CORS | 6.0 | Enables Cross-Origin Resource Sharing so the frontend can call the API |
| Scikit-Learn | 1.8 | Machine learning library used to train Random Forest and Gradient Boosting models |
| NumPy | 2.4 | Numerical computing library used for dataset generation |
| Pandas | 3.0 | Data manipulation library used to structure the training dataset |
| Joblib | 1.5 | Used to save and load trained ML models to/from disk |
| Git | 2.x | Version control to track code changes |
| GitHub | — | Remote repository hosting and CI/CD trigger |
| Docker | 24.x | Containerisation tool to package the app and all dependencies |
| GitHub Actions | — | CI/CD pipeline that automatically runs tests and builds Docker image on every push |
| AWS EC2 | t2.micro | Cloud server where the application is deployed and accessible from the internet |
| pytest | 8.x | Testing framework used to write and run automated tests |

## 4. Local Setup Instructions
Follow these steps to clone the repository and run the application locally using Docker.

### Prerequisites
- Ubuntu or any Linux system
- Docker installed
- Git installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/sehrishnoor/profileguard.git
cd profileguard
```

### Step 2 — Install Docker if not already installed
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 3 — Build the Docker image
```bash
sudo docker build -t profileguard:v1 .
```
This will take 2–3 minutes on the first run as it installs all Python dependencies inside the container.

### Step 4 — Run the container
```bash
sudo docker run -d -p 5000:5000 --name profileguard profileguard:v1
```

### Step 5 — Wait for models to train
On the first run, the ML models need to be trained (takes about 60–90 seconds). Watch the logs:
```bash
sudo docker logs -f profileguard
```
Wait until you see:
```
✅ Models saved.
🚀 Starting Fake Profile Detector API...
* Running on http://0.0.0.0:5000
```

### Step 6 — Open in browser
```
http://localhost:5000
```

### Step 7 — Test the health endpoint
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{"status": "ok", "model": "RandomForest + GradientBoosting Ensemble"}
```

### Step 8 — Stop the container when done
```bash
sudo docker stop profileguard
sudo docker rm profileguard
```

## 5. CI/CD Pipeline Explanation
The CI/CD pipeline is defined in `.github/workflows/ci.yml` and is triggered automatically on every push to the `main` branch.

### What triggers it
Every time code is pushed to the `main` branch on GitHub, the pipeline starts automatically.

### Job 1 — test
This job runs on a fresh Ubuntu environment provided by GitHub Actions. It performs the following steps:
1. Checks out the code from the repository
2. Sets up Python 3.11
3. Installs all dependencies from `requirements.txt`
4. Runs all pytest tests using `python -m pytest test_app.py -v`
If any test fails, this job fails and the second job does not run.

### Job 2 — build-docker
This job only runs if the `test` job passes (controlled by `needs: test`). It performs the following steps:
1. Checks out the code
2. Builds the Docker image using the Dockerfile
3. Starts a container from the built image
4. Waits 10 seconds for the app to start
5. Runs a health check using `curl` against `http://localhost:5000/api/health`
6. Stops the test container

### What happens if a test fails
If any pytest test fails, the `test` job shows a red cross in the GitHub Actions tab and the `build-docker` job is skipped entirely. No broken image is ever built. The developer must fix the failing test and push again before the pipeline goes green.

## 6. Deployment Steps
These are the exact steps taken to deploy the application on AWS EC2.

### Step 1 — Launch EC2 Instance on AWS
1. Logged into AWS Console and went to EC2
2. Clicked Launch Instance
3. Set name: `profileguard-server`
4. Selected OS: Ubuntu Server 22.04 LTS
5. Selected instance type: t2.micro (free tier)
6. Created a new key pair named `profileguard-key`, downloaded the `.pem` file
7. Under Network Settings, added two inbound rules:
   - SSH — Port 22 — Source: Anywhere
   - Custom TCP — Port 5000 — Source: Anywhere
8. Clicked Launch Instance

### Step 2 — Connect to EC2 via SSH
```bash
chmod 400 ~/Downloads/profileguard-key.pem
ssh -i ~/Downloads/profileguard-key.pem ubuntu@16.171.253.13
```

### Step 3 — Update the server and install Docker
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 4 — Clone the repository
```bash
git clone https://github.com/sehrishnoor/profileguard.git
cd profileguard
```

### Step 5 — Build the Docker image
```bash
sudo docker build -t profileguard:v1 .
```

### Step 6 — Run the container with restart policy
```bash
sudo docker run -d -p 5000:5000 --restart=always --name profileguard profileguard:v1
```
The `--restart=always` flag ensures the container automatically restarts if it crashes or if the EC2 instance is rebooted.

### Step 7 — Verify the container is running
```bash
sudo docker ps
```

### Step 8 — Test from the EC2 server itself
```bash
curl http://localhost:5000/api/health
```

### Step 9 — Test from local machine
```bash
curl http://16.171.253.13:5000/api/health
```

### Step 10 — Open frontend in browser
```
http://16.171.253.13:5000
```

## 7. Testing Evidence
### Pytest Tests Passing (Local)
All 5 tests pass locally:
```bash
python3 -m pytest test_app.py -v
```

```
PASSED test_app.py::test_health_check
PASSED test_app.py::test_sample_endpoint
PASSED test_app.py::test_predict_real_profile
PASSED test_app.py::test_predict_fake_profile
PASSED test_app.py::test_predict_missing_fields
5 passed in 45.32s
```

### GitHub Actions Pipeline Green
Both jobs show green checkmarks in the GitHub Actions tab:
- ✅ test — all pytest tests passed
- ✅ build-docker — Docker image built and health check passed

### Live App on EC2
Health check from local machine:
```bash
curl http://16.171.253.13:5000/api/health
```
Response:
```json
{
  "model": "RandomForest + GradientBoosting Ensemble",
  "status": "ok"
}
```
POST request to predict endpoint:
```bash
curl -X POST http://16.171.253.13:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"followers_count":14,"following_count":4980,"post_count":3,
       "account_age_days":12,"has_profile_pic":0,"bio_length":0,
       "has_website_link":0,"is_verified":0,"avg_posts_per_week":0.2,
       "avg_likes_per_post":1,"avg_comments_per_post":0,
       "avg_shares_per_post":0,"engagement_rate":0.001,
       "reply_to_comments_rate":0,"pct_original_posts":0,
       "posting_hour_variance":0.3,"mutual_friends_ratio":0,
       "spam_report_count":5,"login_location_changes":15,
       "username_digit_ratio":0.62,"username_length":22,
       "name_matches_username":0}'
```
Response:
```json
{
  "label": "Fake",
  "trust_score": 3,
  "fake_probability": 97.0,
  "real_probability": 3.0,
  "confidence": 97.0,
  "risk_flags": [
    "Account is very new (< 30 days)",
    "Extreme following/follower imbalance",
    "Multiple spam reports",
    "Very low engagement rate",
    "Username contains many digits (bot-like)",
    "No profile picture",
    "No or very short bio",
    "Suspicious login location changes",
    "Mostly non-original content (reposts/spam)"
  ]
}
```

## 8. Challenges and Solutions

### Challenge 1 — Flask only accessible on localhost inside Docker
After building and running the Docker container, the health check kept returning `curl: (56) Recv failure: Connection reset by peer`. The app was running inside the container but not reachable from outside.

**Root Cause:** In `app.py`, the Flask server was started with `app.run(debug=True, port=5000)` which defaults to binding on `127.0.0.1` (localhost only). Inside a Docker container, localhost means only inside the container itself, so no external connections could reach it.

**Solution:** Changed the last line of `app.py` to:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```
Setting `host='0.0.0.0'` tells Flask to listen on all network interfaces, allowing Docker to forward external connections into the container.

### Challenge 2 — curl returning "Not Found" when accessing EC2 public IP
After deploying to EC2 and confirming the container was running, opening `http://16.171.253.13:5000` in the browser showed a "Not Found" error. The API endpoints worked fine but the root URL had no route.

**Root Cause:** Flask only serves routes that are explicitly defined. There was no route defined for `/` so Flask returned a 404 Not Found for the root URL.

**Solution:** Added a route to `app.py` to serve the frontend HTML file:
```python
from flask import send_from_directory

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')
```
After rebuilding the Docker image and redeploying, the frontend loaded correctly at the root URL.

### Challenge 3 — Models training every time container starts
Every time the Docker container started, it trained the ML models from scratch which took 60–90 seconds. During this time the health endpoint was unreachable.

**Root Cause:** The `.dockerignore` file excluded `*.pkl` files so the pre-trained model files were never copied into the Docker image. The app therefore always found no saved models and triggered full retraining on every start.

**Solution:** Removed `*.pkl` from `.dockerignore` so the pre-trained model files are included in the Docker image. After rebuilding, the container starts in under 5 seconds by loading the saved models instead of retraining.

## 9. Lessons Learned
1. **Docker networking is different from localhost networking.** Before this project I assumed that if something runs on port 5000 it is automatically accessible. I learned that inside a Docker container, `127.0.0.1` is isolated to that container only and you must explicitly bind to `0.0.0.0` to allow traffic in from the host machine or the internet.

2. **CI/CD pipelines catch mistakes before they reach production.** During the project my GitHub Actions pipeline failed twice because of import errors in the test file. Without the pipeline I would have pushed broken code to the server. Having automated tests run on every push meant I caught problems immediately.

3. **The order of Dockerfile instructions matters for build speed.** I learned that copying `requirements.txt` and running `pip install` before copying the rest of the application code means Docker caches the dependency layer. If I only change `app.py`, Docker skips reinstalling all packages and the rebuild is much faster.

4. **AWS Security Groups are like a firewall and must be configured correctly.** My app was running perfectly on the server but unreachable from the browser until I added port 5000 to the inbound security group rules. The EC2 instance was silently blocking all traffic on that port.

5. **Reading error messages carefully saves a lot of time.** Most of the problems I faced were explained clearly in the error output — for example `Running on http://127.0.0.1:5000` told me exactly why external connections were failing. I learned to read the full error message before searching online.

*Document written by: Maida Fatima, Mian Rabia Ilyas, Sehrish Noor, Tooba Hussain*  
*Project: SE202L — Development Operations Lab*  
*Date: May 2026*
Url Preview for GitHub - sehrishnoor/profileguard
GitHub - sehrishnoor/profileguard

Contribute to sehrishnoor/profileguard development by creating an account on GitHub.
github.com
