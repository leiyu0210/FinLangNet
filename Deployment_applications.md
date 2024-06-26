### 1. Application Scenarios

The lifecycle of credit management can broadly be divided into three stages: pre-loan (admission review, credit line allocation, disbursement approval, etc.), during loan (credit line management, churn prediction, marketing response, etc.), and post-loan (collections: repayment rate prediction, aging roll rate, lost contact prediction). Our model is utilized for Behavior Scoring during the loan phase in cash loan services. The introduction of the Behavior Score (B-Score) aims to dynamically monitor risk fluctuations after disbursement. Typically, real-time deployment of the B-Score is not necessary; offline computation on a T+1 basis suffices. This scenario was chosen due to the availability of ample feature sources from existing customers, who have sufficient loan application and repayment records, facilitating our behavior-based modeling approach.

### 2. Application Methodology

Considering the demands for stability and interpretability of model scores in downstream credit risk control strategies, our deep learning-based sub-score is used as a feature in the training of an XGBoost model, contributing to a hybrid tree model. This sub-score concept is fairly common among various fintech companies. Our process is divided into several steps: 

(1) As it is an offline model, concerns about the QPS pressure on online servers are moot. We generate features daily for the entire business's existing customer base, which encompasses millions of users, using dedicated GPU machines. The model inference results are stored for downstream utilization. Meanwhile, we implement daily monitoring to ensure the reliability of production outcomes, such as overseeing the overall average score of the customer groups.

(2) We employ the deep learning sub-score alongside the original XGBoost features to train a blended tree model. Other than including the deep learning sub-score, all parameters remain unchanged. This approach allows us to obtain a model where the deep learning sub-score consistently ranks highest among features based on its IV value and importance scores, aligning with our expectations. The model's benefit is measured by observing improvements in the best KS (Kolmogorov-Smirnov statistic), gains in KS by customer segment and age, lift in different bins, and relative improvement in delinquency rates at the top and bottom bins through swap analysis.

(3) Our model has been deployed online in 2024 for mid-loan business operations servicing millions of users. To ensure stability, we monitor vintage curves and bin delinquency rates weekly, alongside daily PSI (Population Stability Index) monitoring of features and output scores. The blended model demonstrates a 1.5 percentage point increase in the best KS compared to the base model, with stable gains observed across various customer segments.

### 3. Swap Set Analysis

| XGBoost\XGBoost+FinLangNet       | (-inf, 472.0] | (472.0, 569.0] | (569.0, 651.0] | (651.0, 756.0] | (756.0, inf] | Number of overdue samples | Number of sample | Percentage samples | Percentage overdue | Passing rate | Overdue rate as a percentage of the passing sample |
|--------------------------|---------------|----------------|----------------|----------------|--------------|---------------------------|------------------|--------------------|--------------------|--------------|------------------------------------------------------|
| **(-inf, 473.0]**        | 39.16%         | 22.95%         | 16.41%         | 12.98%         | 0.00%        | 232438                    | 658086           | 0.200009            | 35.32%             | 100.00%      | 16.11%                                               |
| **(473.0, 557.0]**       | 27.49%         | 19.54%         | 14.50%         | 10.38%         | 8.10%        | 130362                    | 659658           | 0.200487            | 19.76%             | 80.00%       | 11.30%                                               |
| **(557.0, 634.0]**       | 24.35%         | 17.39%         | 12.82%         | 9.18%          | 5.51%        | 88430                     | 663629           | 0.201693            | 13.33%             | 59.95%       | 8.47%                                                |
| **(634.0, 731.0]**       | 20.92%         | 15.39%         | 11.41%         | 7.78%          | 4.87%        | 54188                     | 653209           | 0.198527            | 8.30%              | 39.78%       | 6.01%                                                |
| **(731.0, inf]**         | 33.33%         | 12.54%         | 9.71%          | 6.30%          | 3.16%        | 24542                     | 655703           | 0.199285            | 3.74%              | 19.93%       | 3.74%                                                |
| **Number of overdue samples** | 240396 | 129771 | 85344 | 51780 | 22669 | N/A | N/A | N/A | N/A | N/A | N/A |
| **Number of samples**     | 661833        | 659743        | 658462        | 656056        | 654191       | N/A                         | N/A              | N/A                 | N/A                | N/A          | N/A                                                  |
| **Percentage samples**    | 0.201148       | 0.200512       | 0.200123       | 0.199392       | 0.198825     | N/A                         | N/A              | N/A                 | N/A                | N/A          | N/A                                                  |
| **Percentage overdue**    | 36.32%         | 19.67%         | 12.96%         | 7.89%          | 3.47%        | N/A                         | N/A              | N/A                 | N/A                | N/A          | N/A                                                  |
| **Passing rate**          | 100.00%        | 79.89%         | 59.83%         | 39.82%         | 19.88%       | N/A                         | N/A              | N/A                 | N/A                | N/A          | N/A                                                  |
| **Overdue rate as a percentage of the passing sample** | 16.11% | 11.02% | 8.12% | 5.68% | 3.47% | N/A | N/A | N/A | N/A | N/A | N/A |
| **Relatively elevated delinquency rates** | **2.84%** |  |  |  | **-7.42%** |  |  |  |  |  |  |

**Relatively elevated delinquency rates = (Percentage overdue(XGBoost + FinLangNet) - Percentage overdue(XGBoost)) / Percentage overdue(XGBoost)**

### Overall Swap Set Analysis for XGBoost vs. XGBoost + FinLangNet Models

#### Objective

The analysis aims to compare the classification and delinquency rate changes between the XGBoost model and the XGBoost model integrated with FinLangNet after frequency-based binning. By integrating overall analysis with specific segments’ `Relatively elevated delinquency rates`, we evaluate the enhancement in risk assessment capabilities provided by the inclusion of FinLangNet.

#### Binning Comparison

- **XGBoost**:
    - (-inf, 473.0]
    - (473.0, 557.0]
    - (557.0, 634.0]
    - (634.0, 731.0]
    - (731.0, inf]

- **XGBoost + FinLangNet**:
    - (-inf, 472.0]
    - (472.0, 569.0]
    - (569.0, 651.0]
    - (651.0, 756.0]
    - (756.0, inf]

#### Overall Swap Set Analysis

1. **Model Comparison**:
    - The integration of FinLangNet introduces changes in binning thresholds, indicating alterations in how the model perceives data distribution.
    - Such adjustments in binning reflect direct influences on pass rates and delinquency rates, suggesting that similar customers might be classified into different risk levels with the new model setup.

2. **Differences in Risk Identification**:
    - Adjustments in bin thresholds and observed metric variations hint that FinLangNet likely improved the model’s capability to discern between potential risk and creditworthy features.

#### Analysis with Relatively Elevated Delinquency Rates

- **Bin (-inf, 472.0]**:
    - An increase of 2.84% in the relative delinquency rate indicates higher assessed risk for customers within this score segment by the XGBoost+FinLangNet model.
    - This suggests that the new model has enhanced sensitivity in identifying potential high-risk customers within this range, which is crucial for early risk detection and prevention.

- **Bin (756.0, inf]**:
    - A decrease of -7.42% in the relative delinquency rate reflects lower assessed risks for high-score customers by the new model.
    - It demonstrates that for creditworthy customers, FinLangNet has improved XGBoost’s risk differentiation ability, likely through more detailed feature analysis and learning, achieving more accurate identification of low-risk customers.

### Conclusions and Recommendations

1. **Enhancing Precision in High-Risk Identification**: With the improved predictive performance in the lower score segment (-inf, 472.0], financial institutions should leverage this advantage by adjusting credit policies and reinforcing risk control measures to reduce the rate of bad loans.

2. **Optimizing Services for Low-Risk Customers**: The decrease in delinquency rates in the high-score segment (756.0, inf] indicates that the new model is better at distinguishing genuinely low-risk customers. Financial institutions could consider offering these customers more favorable loan conditions or intensifying marketing efforts to attract and retain these high-quality clients.

3. **Continuous Monitoring and Model Refinement**: Although the XGBoost+FinLangNet model shows improved performance in certain aspects, the long-term effectiveness and stability of the model require ongoing monitoring. Further refining the model and adjusting the binning strategy could help more accurately assess the real risk of different customer groups.
