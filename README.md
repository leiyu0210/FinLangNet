# FinLangNet: A Novel Deep Learning Framework for Credit Risk Prediction Using Linguistic Analogy in Financial Data
<img src="pic/logo.png" alt="Didi" title="Didi">

### See the Deployment Applications section: Deployment_applications.md
The code for the relationship between the processed input data and the language structure is detailed in: **test_data_sample.ipynb**

We opened up some of our sample data: **data_sample.json**

The Performance Metrics across Different Models for Multiple Labels:

| Model  +DeepFM | dob45dpd7 AUC/KS/Gini | **dob90dpd7 AUC/KS/Gini** | dob90dpd30 AUC/KS/Gini | dob120dpd7 AUC/KS/Gini | dob120dpd30 AUC/KS/Gini | dob180dpd7 AUC/KS/Gini | dob180dpd30 AUC/KS/Gini |
|-------------|-----------------------|---------------------------|------------------------|------------------------|-------------------------|------------------------|------------------------|
| LSTM        | 0.7780/0.4093/0.5560  | 0.7273/0.3286/0.4546       | 0.7610/0.3809/0.5221   | 0.7101/0.3021/0.4203   | 0.7362/0.3433/0.4725    | 0.6927/0.2776/0.3854   | 0.7098/0.3043/0.4196   |
| GRU         | 0.7756/0.4041/0.5513  | 0.7259/0.3240/0.4518       | 0.7568/0.3716/0.5136   | 0.7093/0.3005/0.4185   | 0.7337/0.3357/0.4674    | 0.6906/0.2744/0.3813   | 0.7062/0.2975/0.4123   |
| Stack GRU   | 0.7647/0.3844/0.5294  | 0.7233/0.3235/0.4466       | 0.7580/0.3771/0.5160   | 0.7071/0.3002/0.4142   | 0.7348/0.3416/0.4697    | 0.6893/0.2740/0.3785   | 0.7062/0.2995/0.4124   |
| GRU_Attention   | 0.7745/0.4040/0.5491  | 0.7262/0.3274/0.4523       | 0.7610/0.3816/0.5221   | 0.7088/0.3017/0.4176   | 0.7367/0.3444/0.4735    | 0.6914/0.2745/0.3828   | 0.7098/0.3030/0.4195   |
| Transformer | 0.7725/0.3982/0.5450  | 0.7254/0.3262/0.4508       | 0.7595/0.3798/0.5191   | 0.7097/0.3012/0.4194   | 0.7376/0.3454/0.4752    | 0.6930/0.2782/0.3859   | 0.7119/0.3067/0.4238   |
| FinLangNet  | 0.7765/0.4073/0.5529  | **0.7299/0.3334/0.4598**   | 0.7635/0.3865/0.5269   | 0.7140/0.3091/0.4279   | 0.7413/0.3516/0.4826    | 0.6971/0.2851/0.3942   | 0.7157/0.3138/0.4313   |


The tables summarize the results from ablation experiments on the FinLangNet model's different modules, focusing on the dob90dpd7 label:

| FinLangNet Module                                   | AUC    | KS     | GINI   |
|-----------------------------------------|--------|--------|--------|
| Base           | 0.7299 | **0.3334** | 0.4598 |
| Without Multi-Head           | 0.7266 | 0.3282 | 0.4532 |
| Without DependencyLayer      | 0.7303 | 0.3326 | 0.4606 |
| Without Summary ClS                          | 0.7265 | 0.3279 | 0.4531 |
| Without Feature ClS                          | 0.7278 | 0.3299 | 0.4556 |
| Without Summary ClS and Feature ClS         | 0.7254 | 0.3262 | 0.4508 |

This combined format efficiently illustrates how the different settings impact the performance:

| Learning Rate | Bins | Label     | AUC    | KS     | GINI   |
|---------------|------|-----------|--------|--------|--------|
| 0.0005        | 32   | dob90dpd7 | 0.7252 | 0.3259 | 0.4505 |
| 0.0005        | 8    | dob90dpd7 | 0.7299 | **0.3334** | 0.4598 |
| 0.0005        | 16   | dob90dpd7 | 0.7298 | 0.3331 | 0.4597 |
| 0.0005        | 64   | dob90dpd7 | 0.7223 | 0.3231 | 0.4446 |
| 0.001         | 8    | dob90dpd7 | 0.7236 | 0.3227 | 0.4471 |


