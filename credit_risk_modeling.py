#!/usr/bin/env python
# coding: utf-8

# # 1. Load and exploring the data

# In[1]:


# import pandas
import pandas as pd              
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)

# define support function
def inspect_data(df, col=None, n_rows=5):
    # check data shape
    print(f'data shape: {df.shape}')
    
    # define columns
    if col is None:
        col = df.columns
    
    # check data head, use display function to display dataframe
    display(df[col].head(n_rows))
    
def check_missing(df, cut_off=0, sort=True):
    freq=df.isnull().sum()
    percent=df.isnull().sum()/df.shape[0]*100
    types=df.dtypes
    unique=df.apply(pd.unique).to_frame(name='Unique Values')['Unique Values']
    unique_counts = df.nunique(dropna=False)
    df_miss=pd.DataFrame({'missing_percentage':percent,'missing_frequency':freq,'types':types,'count_value':unique_counts,
                          'unique_values':unique})
    if sort:df_miss.sort_values(by='missing_frequency',ascending=False, inplace=True)
    return df_miss[df_miss['missing_percentage']>=cut_off]

# load file loan_data_2007_2014.csv to DataFrame given df as variable
df = pd.read_csv('loan_data.csv')

# inspect data
inspect_data(df)


# In[2]:


# check missing value
check_missing(df)


# In[3]:


#describe dataframe
df.describe()


# In[4]:


# checking for duplicated data
df.duplicated().any()


# In[5]:


df.id.nunique()


# In[6]:


df.member_id.nunique()


# # 2. Defining the label and making target column
# loan_status column will be our label to make the target with:
# - 'good' label will be 'Fully Paid', and 'Does not meet the credit policy. Status:Fully Paid' loan status
# - row with 'Current' loan status will be dropped since we couldn't determine yet wether it is bad or good
# - other loan status will be labeled 'bad'

# In[7]:


# Check value for loan_status
df.loan_status.value_counts()


# In[8]:


df.loan_status.value_counts(normalize=True)*100


# In[9]:


# dropping row with 'Current' value in loan_status column
df = df[df['loan_status'] != 'Current']


# In[10]:


# import numpy
import numpy as np

# define conditions
conditions = [
    (df['loan_status'].isin(['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])),
    (df['loan_status'].isin(['Charged Off', 'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)', 
                             'Default', 'Does not meet the credit policy. Status:Charged Off']))
]

# define corresponding values good and bad (categorical)
values = ['good','bad']

# create the new column based on conditions
df['loan_label'] = np.select(conditions, values, default='Unknown')


# In[11]:


df.shape


# In[12]:


df['loan_label'].value_counts(normalize=True)*100


# In[13]:


df.loan_label.value_counts()


# # 3. Feature engineering and selection
# in this section if needed we will :
# - drop column who have high threshold of missing value (20% or more)
# - drop column who consist all unique value
# - drop column than only have 1 unique value
# - drop column that contain free text value
# - drop based target column(loan_status column)
# - drop column with 1 dominant category
# - drop numerical column that have high corelation with other column (excluding the target column)
# - drop categorical column with high cardinality
# - drop other column we won't be using in the model (expert judgment)
# - manipulate value on features including handling missing value

# In[14]:


# drop column who have high threshold of missing value (20% or more)
threshold = len(df) * 0.8
df = df.dropna(axis=1, thresh=threshold)

# drop column who have all unique value
unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
df = df.drop(columns=unique_cols)

# drop column with only 1 unique value
single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=single_value_cols)

# drop free text value column
#df = df.drop(columns=['desc'])
# have been dropped due to high missing value

# drop based target column
df = df.drop(columns=['loan_status'])


# In[15]:


# checking column with 1 dominant category, 80% will be the threshold
for col in df.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = df[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        print(value_counts_percentage)
        print('\n')


# In[16]:


# drop column with 1 dominant category automaticaly, 80% will be the threshold
for col in df.select_dtypes(include='object').columns.tolist():
    value_counts_percentage = df[col].value_counts(normalize=True) * 100
    if any(value_counts_percentage > 80):
        df = df.drop(columns=col)


# In[17]:


# checking the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr()

# create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 6})
plt.title('Correlation Heatmap')
plt.show()


# In[18]:


# drop column that have high corelation with other column (correlation coefficient >0.5 or <-0.5), choose 1 column to stay
df = df.drop(columns=['funded_amnt','funded_amnt_inv','installment','total_pymnt','total_pymnt_inv','total_rec_prncp',
                      'total_rec_int','last_pymnt_amnt','collection_recovery_fee','out_prncp_inv','open_acc'])


# In[19]:


# checking the correlation matrix
correlation_matrix = df.corr()

# create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', center=0, annot_kws={'size': 8})
plt.title('Correlation Heatmap')
plt.show()


# In[20]:


df.shape


# In[21]:


check_missing(df)


# In[22]:


# at this point we have 32 features, i'm going to set it as threshold for determining high cardinality
# drop categorical column with high cardinality (unique value count > 32) and expert judgement (i.e : sub_grade will be dropped)
df = df.drop(columns=['emp_title','last_pymnt_d','earliest_cr_line','last_credit_pull_d','title','addr_state','zip_code',
                      'issue_d','sub_grade'])


# In[23]:


df.shape


# In[24]:


check_missing(df)


# In[25]:


# handling missing value
# defining columns with missing value
categorical_columns = ['emp_length']
numerical_columns = ['revol_util','collections_12_mths_ex_med','inq_last_6mths','acc_now_delinq','delinq_2yrs','total_acc',
                     'pub_rec','annual_inc']

# impute categorical columns with mode because it helps to preserve the overall distribution of categorical data and is 
# suitable when the missing values are expected to occur at random.
for col in categorical_columns:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

# impute numerical columns with median because it less sensitive to extreme values (outliers)
for col in numerical_columns:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)


# In[26]:


check_missing(df)


# # 4. Feature scaling and encoding
# - Numerical columns will be scaled using StandardScaler
# - Categorical columns will be encoded using OneHotEncoding
# - Developing dataframe for machine learning modeling

# In[27]:


# define categorical columns
categorical_cols = [col for col in df.select_dtypes(include='object').columns.tolist()]

# develop onehot encoding dataframe
onehot = pd.get_dummies(df[categorical_cols], drop_first=True)

# drop the target columns
onehot = onehot.drop(columns=['loan_label_good'])


# In[28]:


# check onehot encoding dataframe
onehot.head()


# In[29]:


onehot.shape


# why we use onehotencoding for categorical column:
# - Preventing Ordinal Relationships: One-hot encoding eliminates any ordinal relationship that may be incorrectly inferred by the algorithm from the original categorical values. In other words, it treats each category as independent and avoids introducing unintended ordinal relationships.
# - Handling Nominal Categories: For nominal categorical variables (categories without inherent order), one-hot encoding is particularly useful. It ensures that all categories are treated equally and prevents the algorithm from misinterpreting the nominal categories as having an ordinal relationship.
# - Avoiding Misinterpretation as Numeric Values: Without one-hot encoding, some algorithms may incorrectly interpret categorical variables with numeric labels as having a meaningful numeric relationship. One-hot encoding avoids this misinterpretation.

# In[30]:


# import library
from sklearn.preprocessing import StandardScaler

# define numerical columns
numerical_cols = [col for col in df.columns.tolist() if col not in categorical_cols + ['loan_label']]

# develop standardscaler dataframe
ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(df[numerical_cols]), columns=numerical_cols)


# In[31]:


# check standardscaler dataframe
std.head()


# In[32]:


std.shape


# why we use standardscaler for numerical columns:
# - Normalization of Scale: StandardScaler transforms the numerical features in such a way that they have a mean of 0 and a standard deviation of 1. This brings all the features to a common scale, preventing features with larger scales from dominating those with smaller scales.
# - Improving Model Convergence: Many machine learning algorithms, especially those that involve gradient descent optimization, converge faster when the input features are on a similar scale. Standardizing features using StandardScaler can accelerate the convergence of these algorithms.
# - Equal Weight to Features: StandardScaler ensures that all numerical features contribute equally to the model's learning process. Without standardization, features with larger scales may have a disproportionately larger impact on the model.
# - Enhancing Interpretability: In some models, interpretability is important. Standardizing features makes it easier to interpret the model coefficients, as they represent the change in the target variable associated with a one-standard-deviation change in the corresponding feature.
# - Supporting Regularization: Regularization techniques, such as L1 or L2 regularization, penalize large coefficients. StandardScaler can help prevent certain features from dominating the regularization term, making regularization more effective.
# - Assuming Normal Distribution: Some machine learning models, such as linear regression, assume that the features are normally distributed. StandardScaler helps to meet this assumption by transforming the features to have a mean of 0 and a standard deviation of 1.

# In[33]:


# resetting the index of each dataframe to make sure the row numbers doesn't add up
df_reset = df.reset_index(drop=True)
onehot_reset = onehot.reset_index(drop=True)
std_reset = std.reset_index(drop=True)

# develop dataframe for machine learning modeling
df_model = pd.concat([onehot_reset, std_reset, df_reset[['loan_label']]], axis=1)


# In[34]:


# check the model dataframe
df_model.head()


# In[35]:


df_model.shape


# # 5. Modeling - train and evaluate
# - Train - test split (20% - 80%)
# - Conducting imbalance resampling only on the train set
# - Developing several models
# - Evaluate the model

# In[36]:


# define features (X) and target variable (Y)
X = df_model
Y = df_model['loan_label']

# get the column names as feature names
feature_names = X.columns.tolist()


# In[37]:


# import library
from sklearn.model_selection import train_test_split

# split the data into training and testing sets (80% training, 20% testing for n between 100,000 to 1,000,000)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[38]:


# check dataframe shape
X_train.shape, X_test.shape


# In[39]:


# count the occurrences of each loan category
loan_category_counts = X_train['loan_label'].value_counts()

# set colors for different loan categories
colors = ['green', 'red']

# plot the distribution of loan label
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Loan Label')
plt.ylabel('Count')
plt.title('Distribution of Loan Label')

# add values on top of the bars
for i, count in enumerate(loan_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

plt.show()


# In[40]:


# import library
from imblearn.over_sampling import RandomOverSampler

# conducting oversampling using randomoversampler because it's simple and easy to implement and less sensitive to noisy data
# test for overfitting will be conducted later
oversample = RandomOverSampler(sampling_strategy = 'not majority')
X_train, Y_train = oversample.fit_resample(X_train, Y_train)


# In[41]:


# count the occurrences of each loan category
loan_category_counts = X_train['loan_label'].value_counts()

# set colors for different loan categories
colors = ['green', 'red']

# plot the distribution of loan label
plt.bar(loan_category_counts.index, loan_category_counts.values, color=colors)
plt.xlabel('Loan Label')
plt.ylabel('Count')
plt.title('Distribution of Loan Label')

# add values on top of the bars
for i, count in enumerate(loan_category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

plt.show()


# In[42]:


# droping target column
X_train = X_train.drop('loan_label', axis=1)
X_test = X_test.drop('loan_label', axis=1)


# In[43]:


# check dataframe shape
X_train.shape, X_test.shape


# In[44]:


# import library
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# initialize different models
results = {}
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

# initialize dictionary to store classification reports
classification_reports = {}
model_names = []
accuracies = []

# train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, Y_train)

    print(f"Evaluating {model_name}...")
    Y_pred = model.predict(X_test)

    confusion = confusion_matrix(Y_test, Y_pred)
    classification_rep = classification_report(
        Y_test, Y_pred, target_names=['Good', 'Bad'], zero_division=1  # handle zero division
    )

    # store the classification report in the dictionary
    classification_reports[model_name] = classification_rep

    accuracy = accuracy_score(Y_test, Y_pred)

    model_names.append(model_name)
    accuracies.append(accuracy)

    print("\nClassification Report:")
    print(classification_rep)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print("=" * 50)


# In[45]:


# create a bar plot to visualize accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color='skyblue')

# add annotations in the middle of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

# set labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')

# set y-axis limits to 0-1 for accuracy percentage
plt.ylim(0.7, 1)

# rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.tight_layout()

# add horizontal axis line at y=0.9
plt.axhline(0.9, color='black', linewidth=0.8)

plt.show()


# we can see that the random forest model give the highes accuracy (0.92) where:
# - f1≥0.9: excellent
# - 0.8≤f1<0.9: very good
# - 0.7≤f1<0.8: good
# - 0.6≤f1<0.7: fair
# - f1<0.6: poor
# 
# we will evaluate the model performance with auc and ks since it is the most common test to evaluate credit risk modeling

# In[46]:


# converting target to numerical so we can evaluate it with auc and ks
Y_train = Y_train.map({'good': 1, 'bad': 0})
Y_train = Y_train.astype(int)

Y_test = Y_test.map({'good': 1, 'bad': 0})
Y_test = Y_test.astype(int)


# In[47]:


# initialize model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, Y_train)

y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(Y_test), columns=['y_actual']), 
                                 pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = Y_test.index


# In[48]:


# import library
from sklearn.metrics import roc_curve, roc_auc_score

# initialize auc
fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[49]:


# define variables for ks
df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[50]:


# innitialize ks
KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# the model result for auc is 0.90 and ks is 0.68 (on credit risk modeling auc > 0.7 and ks > 0.2 is considered a good performance, which the threshold for each test are:
# 
# AUC:
# - auc = 0.5: random classifier (no discrimination)
# - 0.5 < auc < 0.7: poor discrimination
# - 0.7 ≤ auc < 0.8: acceptable discrimination
# - 0.8 ≤ auc < 0.9: excellent discrimination
# - auc ≥ 0.9: outstanding discrimination
# 
# KS:
# - ks = 0: the distributions of positive and negative samples are identical
# - 0 < ks < 0.2: very small difference between distributions
# - 0.2 ≤ ks < 0.5: moderate difference
# - ks ≥ 0.5: considerable difference

# In[51]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()

# specify the number of folds (K)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# perform K-Fold Cross-Validation on train set
scores = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')

# print the accuracy scores for each fold
for i, score in enumerate(scores, 1):
    print(f'Fold {i}: Accuracy = {score:.4f}')

# print the mean and standard deviation of the accuracy scores
print(f'Mean Accuracy: {np.mean(scores):.4f}')
print(f'Standard Deviation: {np.std(scores):.4f}')


# the k fold cross validation result indicate a model with high accuracy (around 98.5%) and low standard deviation, suggesting that the model performs consistently well across different subsets of the data

# In[52]:


# generating feature importances
arr_feature_importances = rfc.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features


# In[53]:


# showing top 10 feature importance on bar chart
df_top_features = df_all_features.head(10).sort_values(by='importance', ascending=True)

# plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(df_top_features['feature'], df_top_features['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')

# annotating the bars with the importance values
for index, value in enumerate(df_top_features['importance']):
    plt.text(value, index, f'{value:.4f}', va='center')

plt.show()

