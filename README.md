Here's a professional and structured README file for your project:

---

# **Credit Card Fraud Detection using Sampling Techniques and Machine Learning Models**

## **Overview**
This project explores various sampling techniques to handle class imbalance in the Credit Card Fraud Detection dataset and evaluates their performance using five different machine learning models. The dataset is highly imbalanced, and the goal is to determine the most effective combination of sampling techniques and machine learning models for detecting fraudulent transactions.

---

## **Dataset**
The dataset is sourced from the following link:  
[Credit Card Data CSV](https://github.com/AnjulaMehto/Sampling_Assignment/blob/f0c491556cded07517283c75e603bccb70112c26/Creditcard_data.csv)

### **Data Description**
- **Target Variable**:  
  - `Class`: Indicates whether a transaction is fraudulent (1) or non-fraudulent (0).
- **Features**:  
  - `V1` to `V28`: Anonymized numerical features representing various characteristics of the transactions.
  - Other features: `Time` and `Amount`.

---

## **Objective**
1. Handle class imbalance using five different sampling techniques:
   - Random Undersampler
   - Random Oversampler
   - Tomek Links Sampling
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - NearMiss
2. Train and evaluate five machine learning models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
3. Compare the performance of each sampling technique and machine learning model combination to identify the best solution for fraud detection.

---

## **Project Workflow**
1. **Data Loading and Exploration**:  
   Load the dataset, check class imbalance, and perform basic data exploration.

2. **Data Preprocessing**:  
   - Normalize features for consistent scaling.
   - Split the data into training and testing sets.

3. **Apply Sampling Techniques**:  
   Use various sampling techniques to balance the dataset:
   - Random Undersampler
   - Random Oversampler
   - Tomek Links
   - SMOTE
   - NearMiss

4. **Train Machine Learning Models**:  
   Train the following models with each sampling technique and evaluate their performance:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

5. **Performance Evaluation**:  
   Compare models using performance metrics such as accuracy, precision, recall, and F1-score to determine the best combination.

---

## **Results**
- The **Random Forest Classifier** combined with the **Random Oversampler** achieved the highest accuracy of **0.997817**.
- The **Support Vector Machine (SVM)** with **SMOTE** sampling also performed well with an accuracy of **0.990170**.

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. Run the main script to reproduce the results:
   ```bash
   python main.py
   ```
2. Modify the sampling technique or machine learning model in the script for custom experimentation.

---

## **Directory Structure**
```
Credit-Card-Fraud-Detection/
│
├── data/
│   └── Creditcard_data.csv
├── notebooks/
│   └── EDA_and_Preprocessing.ipynb
├── src/
│   ├── sampling.py         # Contains code for all sampling techniques
│   ├── models.py           # Machine learning model definitions
│   ├── evaluation.py       # Performance evaluation metrics
├── main.py                 # Main script for running the pipeline
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
```

---

## **Technologies Used**
- **Python**: Core programming language
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`: Data manipulation and visualization
  - `scikit-learn`: Machine learning models
  - `imbalanced-learn`: Sampling techniques

---

## **Contributing**
Contributions are welcome! Please fork the repository and create a pull request for any suggestions or improvements.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgments**
- Dataset provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- Inspired by real-world challenges of handling imbalanced datasets in fraud detection.

--- 

Let me know if you'd like to tailor it further!
