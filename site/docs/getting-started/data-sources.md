# Freely available data sources

This guide provides curated, high-quality public datasets for testing GlassAlpha's auditing capabilities. All datasets are freely available and suitable for fairness, bias detection, and compliance auditing.

## Quick reference

| Dataset                                                   | Domain           | Size    | Difficulty        | Protected Attributes        | Best For                              |
| --------------------------------------------------------- | ---------------- | ------- | ----------------- | --------------------------- | ------------------------------------- |
| [German Credit](#german-credit-built-in)                  | Finance          | 1,000   | ⭐ Beginner       | Gender, age, foreign worker | Credit risk, fair lending (built-in)  |
| [Adult Income](#adult-income-census)                      | Employment       | 48,842  | ⭐⭐ Intermediate | Race, sex, age              | Income prediction, hiring fairness    |
| [COMPAS Recidivism](#compas-recidivism)                   | Criminal Justice | 7,214   | ⭐⭐ Intermediate | Race, sex, age              | Criminal justice, risk assessment     |
| [HELOC Credit](#home-equity-line-of-credit-heloc)         | Finance          | 10,459  | ⭐⭐ Intermediate | Multiple risk factors       | Credit scoring, risk assessment       |
| [Taiwan Credit](#taiwan-credit-default)                   | Finance          | 30,000  | ⭐⭐ Intermediate | Sex, age, education         | Credit default, Asian demographics    |
| [Communities and Crime](#communities-and-crime)           | Criminal Justice | 1,994   | ⭐⭐ Intermediate | Race composition, income    | Crime prediction, socioeconomic bias  |
| [Credit Card Fraud](#credit-card-fraud-detection)         | Finance          | 284,807 | ⭐⭐⭐ Advanced   | Time, amount patterns       | Fraud detection, imbalanced data      |
| [Folktables (ACS)](#folktables-american-community-survey) | Census           | 1M+     | ⭐⭐⭐ Advanced   | Race, sex, age, disability  | Multiple domains, large-scale testing |

**Difficulty levels**:

- ⭐ **Beginner**: Built-in, small size, simple setup - perfect for first audit
- ⭐⭐ **Intermediate**: Manual download, standard size, straightforward configuration
- ⭐⭐⭐ **Advanced**: Large datasets, special handling, or complex preprocessing needed

## Finance and credit

### German Credit (built-in)

**Built into GlassAlpha** - No download required.

- **Source**: UCI Machine Learning Repository
- **Records**: 1,000 loan applications
- **Features**: 20 demographic, financial, and loan characteristics
- **Target**: Binary credit risk (good/bad)
- **Protected Attributes**: Gender, age groups, foreign worker status

**Usage with GlassAlpha:**

```yaml
data:
  dataset: german_credit # Automatically fetched and cached
  fetch: if_missing
  target_column: credit_risk
  protected_attributes:
    - gender
    - age_group
    - foreign_worker
```

**Why use this:**

- Canonical fairness benchmark in ML
- Well-studied for bias detection
- Manageable size for quick testing
- Multiple protected attributes

**Quick start:**

```bash
glassalpha audit --config german_credit_simple.yaml --output audit.pdf
```

### Adult Income (Census)

Predict whether income exceeds $50K/year based on census data.

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **Records**: 48,842 (32,561 training, 16,281 test)
- **Features**: 14 attributes (age, workclass, education, occupation, etc.)
- **Target**: Binary income (>50K or <=50K)
- **Protected Attributes**: Race, sex, age, native-country

**Download:**

```bash
# Training data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# Test data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

**Usage with GlassAlpha:**

See our complete example configuration in `configs/adult_income.yaml`

```yaml
data:
  dataset: custom
  path: ~/data/adult.data
  target_column: income
  protected_attributes:
    - sex
    - race
    - age
    - native-country

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
```

**Why use this:**

- One of the most-studied fairness datasets
- Multiple intersecting protected attributes
- Real-world census data with known biases
- Large enough for robust testing

### COMPAS Recidivism

Criminal risk assessment scores used in the US justice system.

- **Source**: [ProPublica GitHub](https://github.com/propublica/compas-analysis) or [Kaggle](https://www.kaggle.com/danofer/compass)
- **Records**: 7,214 defendants
- **Features**: Criminal history, demographics, risk scores
- **Target**: Two-year recidivism (binary)
- **Protected Attributes**: Race, sex, age

**Download:**

```bash
# Direct download from ProPublica
wget https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

**Usage with GlassAlpha:**

See our complete example configuration in `configs/compas_recidivism.yaml`

```yaml
data:
  dataset: custom
  path: ~/data/compas-scores-two-years.csv
  target_column: two_year_recid
  protected_attributes:
    - race
    - sex
    - age_cat

model:
  type: logistic_regression
  params:
    random_state: 42
    max_iter: 1000

metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
      - equalized_odds
    config:
      demographic_parity:
        threshold: 0.05
```

**Why use this:**

- Real-world high-stakes decision system
- Known racial bias documented by ProPublica
- Criminal justice fairness research benchmark
- Raises important ethical questions

**Notable research:**

- ProPublica's "Machine Bias" investigation (2016)
- Extensive academic literature on COMPAS bias

### Home Equity Line of Credit (HELOC)

Credit risk assessment for home equity lines of credit.

- **Source**: [FICO Community](https://community.fico.com/s/explainable-machine-learning-challenge)
- **Records**: 10,459 real anonymized credit applications
- **Features**: 23 features derived from credit bureau data
- **Target**: Binary good/bad credit performance
- **Protected Attributes**: Not explicitly included (anonymized)

**Download:**

Visit FICO Explainability Challenge page and download `heloc_dataset_v1.csv`

**Usage with GlassAlpha:**

```yaml
data:
  dataset: custom
  path: ~/data/heloc_dataset_v1.csv
  target_column: RiskPerformance
  feature_columns:
    - ExternalRiskEstimate
    - MSinceOldestTradeOpen
    - MSinceMostRecentTradeOpen
    # ... other features

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 150
    max_depth: 6

explainers:
  strategy: first_compatible
  priority:
    - treeshap
  config:
    treeshap:
      max_samples: 1000
      check_additivity: true
```

**Why use this:**

- Real credit data (anonymized)
- FICO-sponsored explainability challenge
- No synthetic features
- Good for TreeSHAP testing

### Taiwan Credit Default

Credit card default prediction from Taiwan banking data.

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Records**: 30,000 credit card clients
- **Features**: 23 features including payment history, demographics
- **Target**: Binary default payment next month
- **Protected Attributes**: Sex, age, education, marriage

**Download:**

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
```

**Usage with GlassAlpha:**

```yaml
data:
  dataset: custom
  path: ~/data/taiwan_credit_default.csv
  target_column: default_payment_next_month
  protected_attributes:
    - SEX
    - AGE
    - EDUCATION
    - MARRIAGE

model:
  type: lightgbm
  params:
    objective: binary
    n_estimators: 100
    num_leaves: 31
```

**Why use this:**

- Asian demographics (less represented in US datasets)
- Multiple protected attributes
- Real banking data
- Good size for testing (30K records)

### Credit Card Fraud Detection

Highly imbalanced fraud detection dataset.

- **Source**: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions over 2 days
- **Features**: 30 (28 PCA-transformed + Time + Amount)
- **Target**: Binary fraud (0.172% fraud rate)
- **Protected Attributes**: Not included (anonymized via PCA)

**Download:**

Requires Kaggle account: https://www.kaggle.com/mlg-ulb/creditcardfraud

**Usage with GlassAlpha:**

See our complete example configuration `credit_card_fraud_template.yaml` (packaged with GlassAlpha)

```yaml
data:
  dataset: custom
  path: ~/data/creditcard.csv
  target_column: Class

model:
  type: xgboost
  params:
    objective: binary:logistic
    scale_pos_weight: 577 # Handle 0.172% fraud rate
    n_estimators: 100
    max_depth: 6

metrics:
  performance:
    metrics:
      - precision # Critical for fraud
      - recall # Catch actual fraud
      - f1
      - auc_roc
```

**Why use this:**

- Real-world highly imbalanced problem
- Test handling of rare events
- Fraud detection use case
- Large dataset (280K+ records)

**Note**: PCA transformation removes interpretability of individual features.

## Census and demographics

### Folktables (American Community Survey)

Modern, large-scale census data with multiple prediction tasks.

- **Source**: [Folktables GitHub](https://github.com/socialfoundations/folktables)
- **Records**: 1M+ from US Census Bureau's American Community Survey
- **Features**: Varies by task (demographics, employment, income, etc.)
- **Target**: Multiple tasks available
- **Protected Attributes**: Race, sex, age, disability status

**Available prediction tasks:**

- **ACSIncome**: Predict income >$50K (similar to Adult)
- **ACSEmployment**: Predict employment status
- **ACSMobility**: Predict whether moved in past year
- **ACSPublicCoverage**: Predict government health insurance
- **ACSTravelTime**: Predict commute time

**Installation:**

```bash
pip install folktables
```

**Download and prepare:**

```python
from folktables import ACSDataSource, ACSIncome

# Download data
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)

# Extract features and target
features, labels, _ = ACSIncome.df_to_pandas(acs_data)

# Save for GlassAlpha
import pandas as pd
df = features.copy()
df['income'] = labels
df.to_csv('acs_income_ca_2018.csv', index=False)
```

**Usage with GlassAlpha:**

See our complete example configuration `folktables_income_template.yaml` (packaged with GlassAlpha)

```yaml
data:
  dataset: custom
  path: ~/data/acs_income_ca_2018.csv
  target_column: income
  protected_attributes:
    - RAC1P # Race
    - SEX
    - AGEP # Age
    - DIS # Disability

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 150
```

**Why use this:**

- Modern census data (more recent than Adult dataset)
- Multiple states and years available
- Multiple prediction tasks
- Large sample sizes
- Active research community

### Communities and Crime

Socioeconomic and crime data for US communities.

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)
- **Records**: 1,994 communities
- **Features**: 128 attributes (demographics, police, crime)
- **Target**: Violent crime per capita
- **Protected Attributes**: Racial composition, income, poverty

**Download:**

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data
```

**Usage with GlassAlpha:**

```yaml
data:
  dataset: custom
  path: ~/data/communities_crime.csv
  target_column: ViolentCrimesPerPop
  protected_attributes:
    - racepctblack
    - racePctWhite
    - racePctAsian
    - racePctHisp
    - medIncome
    - pctWPubAsst

model:
  type: xgboost
  params:
    objective: reg:squarederror # Regression task
    n_estimators: 100
```

**Why use this:**

- Socioeconomic bias testing
- Multiple racial composition features
- Real community-level data
- Controversial domain requiring careful interpretation

**Warning**: This dataset involves sensitive correlations between race and crime. Use carefully and contextualize results appropriately.

## Healthcare

### Diabetes 130-US Hospitals

Hospital readmission prediction for diabetes patients.

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- **Records**: 101,766 hospital admissions
- **Features**: 50 features (demographics, diagnoses, medications)
- **Target**: 30-day readmission
- **Protected Attributes**: Race, gender, age

**Download:**

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
```

**Usage with GlassAlpha:**

```yaml
data:
  dataset: custom
  path: ~/data/diabetic_data.csv
  target_column: readmitted
  protected_attributes:
    - race
    - gender
    - age

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100
    max_depth: 6

metrics:
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
    config:
      demographic_parity:
        threshold: 0.03 # Stricter for healthcare
```

**Why use this:**

- Real healthcare data
- Large sample size (100K+)
- Multiple protected attributes
- High-stakes domain

### MIMIC-III (Clinical Database)

Large-scale critical care database.

- **Source**: [PhysioNet](https://physionet.org/content/mimiciii/)
- **Records**: 58,000+ ICU admissions
- **Features**: 100+ clinical variables
- **Target**: Various (mortality, readmission, diagnoses)
- **Protected Attributes**: Age, gender, ethnicity, insurance

**Access requirements:**

- Complete CITI training course
- Sign data use agreement
- Requires institutional access

**Why use this:**

- Gold standard healthcare dataset
- Comprehensive clinical data
- Multiple prediction tasks
- Real ICU data

**Note**: Requires credentialed access due to HIPAA protections.

## Dataset repositories and aggregators

### UCI Machine Learning Repository

**URL**: https://archive.ics.uci.edu/ml/index.php

- 600+ datasets across all domains
- Well-documented and maintained
- Standard benchmarks for ML research
- Easy download and citation

**Recommended UCI datasets for auditing:**

- Adult Income (classic fairness benchmark)
- German Credit (built into GlassAlpha)
- COMPAS (criminal justice)
- Communities and Crime (socioeconomic)
- Taiwan Credit (international credit)
- Diabetes Hospital (healthcare)

### Kaggle Datasets

**URL**: https://www.kaggle.com/datasets

- 50,000+ public datasets
- Active community contributions
- Competition datasets with baselines
- API for programmatic download

**Recommended Kaggle datasets:**

- Credit Card Fraud Detection
- Home Credit Default Risk
- IEEE-CIS Fraud Detection
- Lending Club Loan Data

**Kaggle API setup:**

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
```

### OpenML

**URL**: https://www.openml.org/

- 20,000+ datasets
- Standardized format and metadata
- Integrated with scikit-learn
- Reproducible experiment tracking

**OpenML with Python:**

```python
from sklearn.datasets import fetch_openml

# Load Adult dataset
adult = fetch_openml('adult', version=2, as_frame=True)
df = adult.frame
df.to_csv('adult_openml.csv', index=False)
```

### IBM AIF360 Datasets

**URL**: https://github.com/Algorithmic-Fairness/AI-Fairness-360

Pre-processed fairness benchmark datasets:

- Adult Income
- German Credit
- COMPAS
- Bank Marketing
- Medical Expenditure

**Installation:**

```bash
pip install aif360
```

**Usage:**

```python
from aif360.datasets import AdultDataset

dataset = AdultDataset()
df = dataset.convert_to_dataframe()[0]
df.to_csv('adult_aif360.csv', index=False)
```

### Fairlearn Example Datasets

**URL**: https://github.com/fairlearn/fairlearn

Microsoft's fairness toolkit includes example datasets:

- Adult Census Income
- Boston Housing (note: controversial due to race feature)

**Installation:**

```bash
pip install fairlearn
```

## Data preparation checklist

Before using external datasets with GlassAlpha:

### 1. Verify data format

- [ ] CSV, Parquet, Feather, or Pickle format
- [ ] Column headers are clear and descriptive
- [ ] No special characters in column names
- [ ] Target column is clearly identified

### 2. Check data quality

- [ ] No excessive missing values (>30% in any column)
- [ ] Categorical variables are properly encoded
- [ ] Numeric variables are in appropriate ranges
- [ ] Date/time fields are parsed correctly

### 3. Identify protected attributes

- [ ] Protected attributes are present in dataset
- [ ] Protected attribute values are consistent
- [ ] Consider intersectional attributes
- [ ] Document any proxies for protected classes

### 4. Understand the domain

- [ ] Read dataset documentation thoroughly
- [ ] Understand data collection methodology
- [ ] Know any biases in data collection
- [ ] Review relevant research papers

### 5. Test with small sample

- [ ] Create small test file (1000 rows)
- [ ] Run quick audit to verify configuration
- [ ] Check for errors in data loading
- [ ] Verify protected attributes are detected

## Example workflow

### Step 1: Download dataset

```bash
# Example: Adult Income dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
mv adult.data adult_income.csv
```

### Step 2: Inspect data

```python
import pandas as pd

df = pd.read_csv('adult_income.csv', header=None)
print(df.head())
print(df.shape)
print(df.dtypes)
```

### Step 3: Add column names

```python
# Adult dataset column names
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
df.columns = columns
df.to_csv('adult_income_processed.csv', index=False)
```

### Step 4: Create GlassAlpha config

```yaml
audit_profile: tabular_compliance

reproducibility:
  random_seed: 42

data:
  dataset: custom
  path: ~/data/adult_income_processed.csv
  target_column: income
  protected_attributes:
    - sex
    - race
    - age

model:
  type: xgboost
  params:
    objective: binary:logistic
    n_estimators: 100

explainers:
  strategy: first_compatible
  priority:
    - treeshap
    - kernelshap

metrics:
  performance:
    metrics:
      - accuracy
      - precision
      - recall
      - f1
      - auc_roc
  fairness:
    metrics:
      - demographic_parity
      - equal_opportunity
```

### Step 5: Run audit

```bash
glassalpha audit \
  --config adult_income_config.yaml \
  --output adult_income_audit.pdf \
  --strict
```

## Best practices

### Data selection

1. **Start with well-studied datasets**: German Credit, Adult Income, COMPAS
2. **Match domain to use case**: Finance, healthcare, criminal justice, etc.
3. **Consider dataset size**: Start small (1K-10K), scale up as needed
4. **Check for known biases**: Read research papers about the dataset

### Protected attributes

1. **Include multiple attributes**: Race, sex, age at minimum
2. **Consider intersectionality**: Race + sex, age + disability
3. **Document missing attributes**: What's not captured in data
4. **Use consistent encoding**: Binary, categorical, or numeric

### Fairness analysis

1. **Set appropriate thresholds**: Stricter for high-stakes domains
2. **Compare multiple metrics**: Demographic parity, equal opportunity, equalized odds
3. **Contextualize results**: What do the numbers mean in practice?
4. **Document limitations**: What biases might be hidden?

### Reproducibility

1. **Document data source**: URL, version, download date
2. **Save preprocessing steps**: Scripts for data cleaning
3. **Use version control**: Git for configs and scripts
4. **Generate manifests**: GlassAlpha's built-in audit trails

## Troubleshooting

### Dataset won't load

**Problem**: `FileNotFoundError` or parsing errors

**Solutions**:

- Check file path is absolute or expandable (`~` for home)
- Verify file format matches extension
- Try reading with pandas first to debug
- Check for special characters in data

### Protected attributes not detected

**Problem**: Fairness metrics show errors

**Solutions**:

- Verify column names match config exactly
- Check for missing values in protected columns
- Ensure protected attributes are in feature set
- Review data types (string vs categorical vs numeric)

### Model training fails

**Problem**: Error during model fitting

**Solutions**:

- Check for missing values (GlassAlpha handles automatically)
- Verify target column has correct binary encoding
- Ensure sufficient samples (>100 minimum)
- Try simpler model first (logistic_regression)

### Large dataset performance

**Problem**: Audit takes too long

**Solutions**:

- Reduce SHAP sample size in config
- Use sample of data for testing
- Enable parallel processing (n_jobs: -1)
- Consider LightGBM over XGBoost (faster)

## Additional resources

### Research papers on fairness datasets

- [Friedler et al. (2019) - "A comparative study of fairness-enhancing interventions"](https://arxiv.org/abs/1802.04422)
- [Hardt et al. (2016) - "Equality of Opportunity in Supervised Learning"](https://arxiv.org/abs/1610.02413)
- [ProPublica (2016) - "Machine Bias"](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

### Tutorials and guides

- [Fairlearn Documentation](https://fairlearn.org/)
- [AIF360 Documentation](https://aif360.readthedocs.io/)
- [Google's ML Fairness Resources](https://developers.google.com/machine-learning/fairness-overview)

### Community resources

- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers with Code - Fairness](https://paperswithcode.com/task/fairness)
- [GitHub Topics - Fairness ML](https://github.com/topics/fairness-ml)

## Ready to Test?

Now that you know what datasets are available, here's your path forward:

1. **🚀 Pick a dataset**: Start with [German Credit](#german-credit-built-in) (easiest, built-in)
2. **📥 Download**: Follow the download instructions for your chosen dataset
3. **⚙️ Configure**: Use the provided GlassAlpha config examples
4. **📊 Run audit**: Generate your first benchmark audit
5. **🔍 Compare**: Try multiple datasets to see how bias varies

**Want to use your own data instead?** [See Using Custom Data](custom-data.md)

**Need help?** Check the [Troubleshooting Guide](../reference/troubleshooting.md) or [FAQ](../reference/faq.md)

**Found this helpful?** [Star us on GitHub ⭐](https://github.com/GlassAlpha/glassalpha) to support the project!

## Contributing datasets

If you have a dataset that would be valuable for the GlassAlpha community:

1. Open an issue on GitHub describing the dataset
2. Provide download link and documentation
3. Share example GlassAlpha configuration
4. Document any known biases or limitations

We're especially interested in:

- Non-US datasets for geographic diversity
- Underrepresented domains (education, housing, etc.)
- Datasets with ground truth fairness assessments
- Large-scale datasets (100K+ records)

---

**Questions or suggestions?** Open an issue on [GitHub](https://github.com/GlassAlpha/glassalpha/issues) or start a [discussion](https://github.com/GlassAlpha/glassalpha/discussions).
