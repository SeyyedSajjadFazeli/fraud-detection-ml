<div align="center">

<h1>💳 Credit Card Fraud Detection System</h1>
<h3>سیستم تشخیص تقلب تراکنش‌های بانکی با یادگیری ماشین</h3>

<p>
An <strong>End-to-End Machine Learning Project</strong> with Streamlit Dashboard, Explainable AI (SHAP),
FastAPI Backend & Docker Support
</p>

<img src="https://img.shields.io/badge/Python-3.9-blue" />
<img src="https://img.shields.io/badge/Model-XGBoost-green" />
<img src="https://img.shields.io/badge/UI-Streamlit-red" />
<img src="https://img.shields.io/badge/API-FastAPI-teal" />
<img src="https://img.shields.io/badge/Deployment-Docker-blue" />

</div>

<hr/>

<p align="center">
<em>Interactive Streamlit dashboard with threshold tuning, auto fraud generation, and live confusion matrix</em>
</p>

<hr/>

<h2>📌 Overview | معرفی پروژه</h2>

<h3>🇺🇸 English</h3>
<p>
This project is an <strong>end-to-end Machine Learning system</strong> designed to detect fraudulent credit card
transactions.
It tackles <strong>extreme class imbalance</strong>, provides an <strong>interactive Streamlit dashboard</strong>,
supports <strong>decision threshold tuning</strong>, includes <strong>Explainable AI (SHAP)</strong>, and offers a
<strong>production-ready FastAPI backend</strong> with Docker support.
</p>

<h3>🇮🇷 فارسی</h3>
<p>
این پروژه یک <strong>سیستم کامل و صنعتی تشخیص تقلب تراکنش‌های بانکی</strong> است که به‌صورت End-to-End پیاده‌سازی شده است.
در این پروژه چالش <strong>نامتوازن بودن شدید داده‌ها</strong> مدیریت شده و امکاناتی مانند داشبورد تعاملی،
تنظیم آستانه تصمیم، تولید داده تقلبی، توضیح‌پذیری مدل و API آماده پروداکشن ارائه شده است.
</p>

<hr/>

<h2>🎯 Problem Statement | مسئله</h2>

<h3>🇺🇸 English</h3>
<p>
Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent
<strong>less than 0.2%</strong> of all data.
Using accuracy alone is misleading; therefore, this project focuses on
<strong>Recall, Precision, ROC-AUC, PR-AUC</strong>, and
<strong>business-driven threshold optimization</strong>.
</p>

<h3>🇮🇷 فارسی</h3>
<p>
تشخیص تقلب کارت‌های بانکی یک مسئله طبقه‌بندی با <strong>عدم توازن شدید کلاس‌ها</strong> است؛
به‌طوری که کمتر از <strong>۰٫۲٪</strong> تراکنش‌ها تقلبی هستند.
در چنین شرایطی معیار Accuracy گمراه‌کننده بوده و تمرکز باید روی
<strong>Recall، Precision و PR-AUC</strong> باشد.
</p>

<hr/>

<h2>🧠 Solution Approach | رویکرد حل مسئله</h2>

<ul>
  <li>Data preprocessing & robust scaling</li>
  <li>Handling class imbalance using <strong>SMOTE</strong></li>
  <li>Training an <strong>XGBoost classifier</strong></li>
  <li>Dynamic decision threshold tuning</li>
  <li>Evaluation using ROC-AUC & PR-AUC</li>
  <li>Explainable AI using <strong>SHAP</strong></li>
  <li>Interactive visualization using <strong>Streamlit</strong></li>
  <li>Production-ready backend using <strong>FastAPI</strong></li>
</ul>

<hr/>

<h2>📊 Model Performance | عملکرد مدل</h2>

<table border="1" cellpadding="8">
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ROC-AUC</td>
    <td>~0.98</td>
  </tr>
  <tr>
    <td>Fraud Recall</td>
    <td>~0.89</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>Threshold-dependent</td>
  </tr>
</table>

<p>
<strong>Note:</strong> High recall is prioritized to minimize missed fraudulent transactions,
which is critical in financial systems.
</p>

<hr/>

<h2>🖥 Streamlit Dashboard | داشبورد تعاملی</h2>

<ul>
  <li>Manual transaction input</li>
  <li>⚖️ Decision threshold slider</li>
  <li>🤖 Auto-generated fraud transactions</li>
  <li>📊 Live confusion matrix</li>
  <li>Real-time fraud probability</li>
</ul>

<pre><code>streamlit run app/app.py</code></pre>

<hr/>

<h2>🧠 Explainable AI (SHAP)</h2>

<p>
Model decisions are explained using <strong>SHAP values</strong> to ensure transparency and trust.
Key features such as <strong>V14, V10, and V17</strong> have the strongest influence on fraud detection.
</p>

<hr/>

<h2>🌐 FastAPI Backend | بک‌اند API</h2>

<pre><code>uvicorn api.main:app --reload</code></pre>

<p>Swagger UI:</p>
<pre><code>http://localhost:8000/docs</code></pre>

<hr/>

<h2>🐳 Docker Support | داکر</h2>

<pre><code>docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api</code></pre>

<hr/>

<h2>📁 Project Structure | ساختار پروژه</h2>

<pre><code>fraud-detection-ml/
├── app/        # Streamlit dashboard
├── src/        # ML pipeline & utilities
├── api/        # FastAPI backend
├── models/     # Trained model
├── data/       # Dataset (not included)
├── requirements.txt
└── README.html
</code></pre>

<hr/>

<h2>📦 Dataset | دیتاست</h2>

<p>
<strong>Kaggle Credit Card Fraud Detection Dataset</strong><br/>
Dataset is not included due to size and license restrictions.
</p>

<hr/>

<h2>🚀 Future Improvements | توسعه‌های آینده</h2>

<ul>
  <li>Model versioning</li>
  <li>Online / incremental learning</li>
  <li>Real-time streaming inference</li>
  <li>Cloud deployment (AWS / GCP)</li>
  <li>Monitoring & logging</li>
</ul>

<hr/>

<h2>👤 Author | توسعه‌دهنده</h2>

<p>
<strong>Seyyed Sajjad Fazeli</strong><br/>
Machine Learning Engineer<br/>
</p>

<hr/>

<div align="center">
⭐ If you find this project useful, consider giving it a star! ⭐
</div>
