### 1. Application Scenarios

The lifecycle of credit management can broadly be divided into three stages: pre-loan (admission review, credit line allocation, disbursement approval, etc.), during loan (credit line management, churn prediction, marketing response, etc.), and post-loan (collections: repayment rate prediction, aging roll rate, lost contact prediction). Our model is utilized for Behavior Scoring during the loan phase in cash loan services. The introduction of the Behavior Score (B-Score) aims to dynamically monitor risk fluctuations after disbursement. Typically, real-time deployment of the B-Score is not necessary; offline computation on a T+1 basis suffices. This scenario was chosen due to the availability of ample feature sources from existing customers, who have sufficient loan application and repayment records, facilitating our behavior-based modeling approach.

### 2. Application Methodology

Considering the demands for stability and interpretability of model scores in downstream credit risk control strategies, our deep learning-based sub-score is used as a feature in the training of an XGBoost model, contributing to a hybrid tree model. This sub-score concept is fairly common among various fintech companies. Our process is divided into several steps: 

(1) As it is an offline model, concerns about the QPS pressure on online servers are moot. We generate features daily for the entire business's existing customer base, which encompasses millions of users, using dedicated GPU machines. The model inference results are stored for downstream utilization. Meanwhile, we implement daily monitoring to ensure the reliability of production outcomes, such as overseeing the overall average score of the customer groups.

(2) We employ the deep learning sub-score alongside the original XGBoost features to train a blended tree model. Other than including the deep learning sub-score, all parameters remain unchanged. This approach allows us to obtain a model where the deep learning sub-score consistently ranks highest among features based on its IV value and importance scores, aligning with our expectations. The model's benefit is measured by observing improvements in the best KS (Kolmogorov-Smirnov statistic), gains in KS by customer segment and age, lift in different bins, and relative improvement in delinquency rates at the top and bottom bins through swap analysis.

(3) Our model has been deployed online in 2024 for mid-loan business operations servicing millions of users. To ensure stability, we monitor vintage curves and bin delinquency rates weekly, alongside daily PSI (Population Stability Index) monitoring of features and output scores. The blended model demonstrates a 1.5 percentage point increase in the best KS compared to the base model, with stable gains observed across various customer segments.

### 3. Swap Set Analysis
