# Major_project
# Customer Churn Predictor

A responsive web-based telecom customer churn prediction system that analyzes customer information and estimates the likelihood of customer churn using a rule-based scoring approach.

---

## Overview

Customer churn refers to customers leaving a service provider. This project predicts churn risk based on customer profile, services, billing information, and usage patterns.

The system classifies customers into:

- 🟢 Low Churn Risk
- 🟡 Medium Churn Risk
- 🔴 High Churn Risk

---

## Features

### Telecom Provider Selection
Users can choose among:

- Reliance Jio
- Airtel
- VI-Idea
- BSNL

### Customer Details

#### Usage & Charges
- Tenure (months)
- Monthly Charges
- Total Charges

#### Customer Profile
- Gender
- Senior Citizen
- Partner
- Dependents

#### Services
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

#### Billing & Contract
- Contract Type
- Payment Method
- Paperless Billing

---

## Technologies Used

- HTML5
- CSS3
- JavaScript

---

## Project Structure

```bash
Customer-Churn-Predictor/
│
├── index.html
├── README.md
└── assets/
```

---

## Prediction Logic

The application calculates a churn score between **0–100**.

### Risk Factors

| Factor | Effect |
|----------|---------|
| Short tenure | Increase risk |
| Month-to-month contract | Increase risk |
| Fiber optic service | Increase risk |
| Electronic check payment | Increase risk |
| Senior citizen | Increase risk |
| No online security | Increase risk |
| No tech support | Increase risk |
| High monthly charges | Increase risk |
| Long tenure | Reduce risk |
| Two-year contract | Reduce risk |

---

## Risk Classification

### High Risk
Score ≥ 65

Recommendation:
Immediate retention action should be taken.

### Medium Risk
Score 35–64

Recommendation:
Monitor and engage customer proactively.

### Low Risk
Score < 35

Recommendation:
Customer appears stable and loyal.

---

## How to Run

1. Download or clone the project:

```bash
git clone https://github.com/your-username/customer-churn-predictor.git
```

2. Open the project folder

3. Run:

```bash
index.html
```

Or use VS Code Live Server extension.

---

## Future Improvements

- Integrate Machine Learning model
- Connect with real telecom datasets
- Add dashboard analytics
- Store prediction history
- Generate PDF reports
- Add backend API integration
- User authentication

---

## Dataset Information

Dataset Summary:

- Total Customers: 7,043
- Features: 31
- Domain: Telecom Customer Churn

---

## Author

Developed for telecom customer churn prediction and analysis using web technologies.
