# Major_project
Customer Churn Predictor
Overview

Customer Churn Predictor is a web-based application designed to analyze telecom customer details and estimate the probability of customer churn (customers leaving a telecom service provider).

The application allows users to enter customer profile information, service usage details, billing information, and telecom partner selection, then predicts churn risk as:

Low Churn Risk
Medium Churn Risk
High Churn Risk

The prediction is based on rule-based scoring logic using customer behavior patterns.

Features
Telecom Partner Selection

Users can select a telecom operator:

Reliance Jio
Airtel
VI-Idea
BSNL

The selected provider is highlighted.

Customer Information
Usage & Charges
Tenure (months)
Monthly charges
Total charges
Customer Profile
Gender
Senior citizen status
Partner status
Dependents
Services
Phone service
Multiple lines
Internet service
Online security
Online backup
Device protection
Tech support
Streaming TV
Streaming movies
Billing & Contract
Contract type
Payment method
Paperless billing
Prediction Logic

The system calculates a churn score from 0–100.

Risk scoring factors:

Factor	Impact
Short tenure	Increase risk
Month-to-month contract	Increase risk
Fiber internet	Increase risk
Electronic check payment	Increase risk
Senior citizen	Increase risk
No online security	Increase risk
No tech support	Increase risk
No partner/dependents	Increase risk
High monthly charges	Increase risk
Long tenure	Reduce risk
Two-year contract	Reduce risk
Risk Levels
High Risk

Score ≥ 65

Description:
Customer has a high chance of leaving. Immediate retention actions are recommended.

Medium Risk

Score between 35–64

Description:
Customer may leave in the future. Proactive engagement is recommended.

Low Risk

Score < 35

Description:
Customer appears loyal and satisfied.

Technologies Used
Frontend
HTML5
CSS3
JavaScript
UI Features
Responsive layout
Custom telecom cards
Dynamic risk visualization
Animated progress bar
Interactive customer forms
Project Structure
Customer-Churn-Predictor/
│
├── index.html
├── README.md
└── assets/
How to Run
Download the project files
Save the code as:
index.html
Open the file in a browser

OR use a local server:

Live Server (VS Code)
Future Improvements
Integrate Machine Learning models
Connect with real telecom datasets
Add charts and analytics dashboards
Store prediction history
Export reports as PDF
Add authentication system
Backend API integration
Dataset Information

Dataset characteristics:

Customers: 7,043
Features: 31
Domain: Telecom Customer Churn Analysis
Author

Developed for Telecom Customer Churn Analysis and Prediction using web technologies.
