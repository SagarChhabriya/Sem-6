**Machine Learning Pipeline**

* **Definition**: A sequence of steps to develop and deploy ML models.
* **Typical Stages**:

  1. **Data Collection**: Gather raw data from sources (databases, CSVs, sensors).
  2. **Data Cleaning & Preprocessing**: Handle missing values, outliers, encoding categorical variables, normalization/standardization.
  3. **Feature Engineering**: Create or select features (e.g., polynomial features, one‐hot encoding).
  4. **Dimensionality Reduction**: Techniques like PCA to reduce feature space.
  5. **Model Selection & Training**: Choose algorithm(s), split into train/validation sets, fit model(s).
  6. **Hyperparameter Tuning**: Grid search or randomized search on validation set.
  7. **Evaluation**: Metrics such as accuracy, precision, recall, F1‐score, ROC‐AUC.
  8. **Deployment**: Serialize model, integrate into application or expose as API.
  9. **Monitoring & Maintenance**: Track performance drift and retrain as needed.

---

## Transfer Learning

* **Definition**: Reusing a pretrained model (often on large datasets) as a starting point for a new task.
* **Key Ideas**:

  * **Feature Reuse**: Lower layers capture general patterns (edges, textures), higher layers specialize.
  * **Fine‐tuning**: Freeze early layers, retrain later layers on your dataset.
  * **Common Use Cases**: Image classification (e.g., using pretrained ResNet, VGG), NLP (e.g., BERT, GPT).
* **Workflow**:

  1. **Choose a Pretrained Base**: e.g., `ResNet50(weights='imagenet')` in Keras.
  2. **Freeze Base Layers**: Prevent weights from updating initially.
  3. **Add New Head**: Dense layers matching your target classes.
  4. **Compile & Train**: On your smaller dataset, often with a lower learning rate.
  5. **Unfreeze & Fine‐Tune** (optional): Unfreeze some base layers and continue training for a few epochs.

---

## Principal Component Analysis (PCA)

* **Goal**: Reduce dimensionality by projecting data onto orthogonal axes (principal components) capturing maximal variance.
* **Steps**:

  1. **Standardize** data (zero mean, unit variance).
  2. **Compute Covariance Matrix** of features.
  3. **Eigen Decomposition**: Find eigenvalues & eigenvectors.
  4. **Sort** eigenvectors by descending eigenvalues.
  5. **Project** data onto top k eigenvectors (components).
* **When to Use**:

  * High‐dimensional data (e.g., hundreds of features).
  * Visualizing in 2D/3D.
  * Preprocessing to remove noise or correlated features.
* **Caveats**: Principal components are linear combinations—interpretability may suffer.

---

## Natural Language Processing (NLP)

* **Core Tasks**:

  * **Tokenization**: Split text into words/subwords.
  * **Stop‐Word Removal**: Remove common words (e.g., “the”, “and”).
  * **Stemming/Lemmatization**: Reduce words to root form.
  * **Vectorization**: Convert tokens to numeric form:

    * **Bag‐of‐Words / TF–IDF**
    * **Word Embeddings** (Word2Vec, GloVe)
    * **Contextual Embeddings** (BERT, RoBERTa)
* **Common Models**:

  * **n-gram Language Models**
  * **Naïve Bayes** (for text classification)
  * **Recurrent Neural Networks** / **LSTM** / **GRU** (for sequence tasks)
  * **Transformer‐based architectures** (BERT, GPT).
* **Pipeline Example**:

  1. Text Ingestion → 2. Cleaning (lowercasing, punctuation removal) → 3. Tokenization → 4. Vectorization → 5. Model (e.g., logistic regression or fine‐tuned BERT).

---

## k-Nearest Neighbors (KNN)

* **Concept**: Classify (or regress) new points based on majority vote (or average) of k closest training examples in feature space.
* **Distance Metric**: Usually Euclidean, but can be Manhattan, Minkowski, cosine, etc.
* **Hyperparameters**:

  * **k**: Number of neighbors.
  * **Distance Metric**.
  * **Weighting**: Uniform (all neighbors equal) or distance‐based.
* **Pros**:

  * Simple to understand & implement.
  * No training phase (instance‐based learning).
* **Cons**:

  * Computation & memory heavy at inference (must compute distances to all points).
  * Sensitive to feature scaling (always scale/normalize first).
  * Curse of dimensionality—performance degrades in high dimensions.

---

## Common Model Algorithms

1. **Linear Regression / Logistic Regression**

   * Linear Regression for continuous targets, Logistic Regression for binary/multiclass classification (using sigmoid/softmax).
2. **Decision Trees**

   * Tree‐structured models splitting on feature thresholds. Interpretability is high; prone to overfitting if deep.
3. **Random Forest & Gradient Boosting**

   * Ensembles of trees:

     * **Random Forest**: Bagging, averaging many trees → reduces variance.
     * **Gradient Boosted Trees** (e.g., XGBoost, LightGBM): Sequentially build trees to correct previous errors → reduces bias.
4. **Support Vector Machines (SVMs)**

   * Find a hyperplane that maximizes margin between classes. Kernels allow nonlinear boundaries.
5. **Neural Networks**

   * Multilayer Perceptrons (MLPs) for tabular data, CNNs for images, RNNs/Transformers for sequences. Highly flexible but require more data & tuning.
6. **Naïve Bayes**

   * Probabilistic classifier assuming feature independence. Commonly used for text classification.

---

## Quick Code Examples (Python)

> **Note**: Install required packages via `pip install numpy pandas scikit-learn tensorflow matplotlib nltk`.

```python
# 1. Machine Learning Pipeline (using sklearn Pipeline)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

# Example dataset (Iris)
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)
X, y = data.data, data.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define pipeline: StandardScaler → PCA → KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('knn', KNeighborsClassifier())
])

# Hyperparameter grid for tuning
param_grid = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

# GridSearch for best params
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
y_pred = grid.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
```

```python
# 2. Transfer Learning (using Keras and a pretrained ResNet50 on ImageNet)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assume you have train_dir and val_dir of image folders structured by class
train_dir = '/path/to/train'
val_dir = '/path/to/val'

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True).flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(224,224), batch_size=32, class_mode='categorical')

# Load base model (without top)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base

# Add new classification head
inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=5, validation_data=val_gen)

# Optional: Unfreeze some layers and fine-tune
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_gen, epochs=3, validation_data=val_gen)
```

```python
# 3. PCA from scratch (using numpy)
import numpy as np

# Toy data (4 samples, 3 features)
X = np.array([
    [2.5, 2.4, 0.5],
    [0.5, 0.7, 1.2],
    [2.2, 2.9, 0.4],
    [1.9, 2.2, 0.3]
])

# 1. Standardize
X_centered = X - np.mean(X, axis=0)

# 2. Covariance matrix
cov_mat = np.cov(X_centered, rowvar=False)

# 3. Eigen decomposition
eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

# 4. Sort eigenvectors
idxs = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[idxs], eig_vecs[:, idxs]

# 5. Project onto top 2 components
W = eig_vecs[:, :2]  # 3×2 matrix
X_pca = X_centered.dot(W)
print("Projected Data:\n", X_pca)
```

```python
# 4. Basic NLP Preprocessing & TF–IDF + Logistic Regression (sklearn)
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample corpus
documents = [
    "Machine learning is fascinating.",
    "I love natural language processing.",
    "Transfer learning helps when data is limited.",
    "PCA reduces dimensions."
]
labels = [0, 1, 1, 0]  # e.g., two classes

# Tokenization & TF–IDF
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b')
X_tfidf = vectorizer.fit_transform(documents)

# Train a simple classifier
clf = LogisticRegression()
clf.fit(X_tfidf, labels)

# Predict on a new sentence
new_text = ["PCA is useful for dimensionality reduction."]
new_vec = vectorizer.transform(new_text)
print("Predicted Class:", clf.predict(new_vec)[0])
```

```python
# 5. k-NN Classification (sklearn)
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = load_wine(as_frame=True)
X_wine, y_wine = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, random_state=0)

# Always scale for k-NN
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

```python
# 6. Example: Common Model Algorithms (Logistic Regression & Decision Tree)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Using Iris again
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
X_train_s = StandardScaler().fit_transform(X_train)
X_test_s = StandardScaler().transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_s, y_train)
print("LR Accuracy:", lr.score(X_test_s, y_test))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print("DT Accuracy:", dt.score(X_test, y_test))
```

---

### Quick Tips for Tomorrow’s Exam

* **Understand the Flow**: Know how each step in a pipeline connects to the next (e.g., why scaling before KNN matters).
* **Key Equations**:

  * **PCA**: Covariance matrix = (1/(n−1)) Xᵀ X; projection = X\_centered · eigenvectors.
  * **KNN Distance**: Euclidean = √Σ(xᵢ − xⱼ)².
* **Terminology**: Be able to define “transfer learning”, “principal components”, “tokenization”, and “weighting in KNN”.
* **Code Recall**: Remember common function names (`PCA(n_components=...)`, `KNeighborsClassifier(n_neighbors=...)`, `TfidfVectorizer(...)`).
* **Hyperparameters**: Explain how to choose k in KNN (cross-validation) or number of components in PCA (variance explained).
* **NLP Preprocessing**: Know difference between stemming vs. lemmatization, and why stop‐word removal helps.
* **Model Algorithms**: Practice naming at least three differences between decision trees vs. logistic regression vs. SVM.

---------------

**Evaluation Metrics (Regression)**

* **Mean Squared Error (MSE)**

  $$
    \text{MSE} = \frac{1}{n}\sum_{i=1}^n \bigl(y_i - \hat{y}_i\bigr)^2
  $$

  Measures average squared error; penalizes larger errors.
* **Root Mean Squared Error (RMSE)**

  $$
    \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
  $$

  Same units as target; easier to interpret.
* **Mean Absolute Error (MAE)**

  $$
    \text{MAE} = \frac{1}{n}\sum_{i=1}^n \bigl|\,y_i - \hat{y}_i\bigr|
  $$

  Linear penalty of errors; less sensitive to outliers.
* **R-squared ($R^2$)**

  $$
    R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $$

  Proportion of variance explained by the model (closer to 1 is better).

---

```python
# k-NN Regressor Example

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. Load dataset
data = load_boston()
X, y = data.data, data.target

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale features
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 4. Train k-NN regressor
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',     # or 'uniform'
    metric='minkowski',     # Euclidean by default (p=2)
    p=2
)
knn_reg.fit(X_train_s, y_train)

# 5. Predict
y_pred = knn_reg.predict(X_test_s)

# 6. Compute evaluation metrics
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
```

---------
**Evaluation Metrics (Classification)**

* **Confusion Matrix**:

  $$
    \begin{array}{cc|cc}
      & & \multicolumn{2}{c}{\text{Predicted}} \\
      & & \text{Positive} & \text{Negative} \\ \hline
      \multirow{2}{*}{\text{Actual}} & \text{Positive} & \text{TP} & \text{FN} \\
      & \text{Negative} & \text{FP} & \text{TN} \\
    \end{array}
  $$

  * TP: True Positives
  * TN: True Negatives
  * FP: False Positives
  * FN: False Negatives

* **Accuracy**

  $$
    \text{Accuracy} 
    = \frac{TP + TN}{TP + TN + FP + FN}
  $$

  Fraction of correctly classified samples.

* **Precision** (a.k.a. Positive Predictive Value)

  $$
    \text{Precision} 
    = \frac{TP}{TP + FP}
  $$

  Of all predicted positives, fraction that are actually positive.

* **Recall (Sensitivity, True Positive Rate)**

  $$
    \text{Recall} 
    = \frac{TP}{TP + FN}
  $$

  Of all actual positives, fraction correctly predicted.

* **Specificity (True Negative Rate)**

  $$
    \text{Specificity} 
    = \frac{TN}{TN + FP}
  $$

  Of all actual negatives, fraction correctly predicted.

* **F1-Score**

  $$
    F_1 
    = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

  Harmonic mean of precision and recall.

* **ROC Curve & AUC**

  * ROC plots TPR vs. FPR (False Positive Rate = FP/(FP+TN)) at various classification thresholds.
  * **AUC (Area Under ROC Curve)**: area under that curve; closer to 1 means better separation.

* **Precision–Recall (PR) Curve & AUC**

  * PR curve plots Precision vs. Recall at different thresholds.
  * **PR-AUC**: area under the PR curve; especially informative for imbalanced data.

* **Logarithmic Loss (Log Loss, Cross-Entropy Loss)**
  For binary classification, if $\hat{p}_i$ is predicted probability of class 1:

  $$
    \text{LogLoss} 
    = -\frac{1}{n} \sum_{i=1}^n \Bigl[y_i \log(\hat{p}_i) 
      + (1 - y_i)\log(1 - \hat{p}_i)\Bigr].
  $$

  Penalizes confident but wrong predictions heavily. Lower is better.

* **Matthews Correlation Coefficient (MCC)**

  $$
    \text{MCC} 
    = \frac{TP \times TN \;-\; FP \times FN}
           {\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}.
  $$

  Balanced measure even if classes are of very different sizes.

---

**Evaluation Metrics (Regression)**

* **Mean Squared Error (MSE)**

  $$
    \text{MSE} 
    = \frac{1}{n}\sum_{i=1}^n \bigl(y_i - \hat{y}_i\bigr)^2.
  $$

  Average squared difference between true and predicted values.

* **Root Mean Squared Error (RMSE)**

  $$
    \text{RMSE} 
    = \sqrt{\text{MSE}} 
    = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}.
  $$

  Same units as the target; easier to interpret.

* **Mean Absolute Error (MAE)**

  $$
    \text{MAE} 
    = \frac{1}{n}\sum_{i=1}^n \bigl|\,y_i - \hat{y}_i\bigr|.
  $$

  Average absolute difference; less sensitive to outliers than MSE.

* **Mean Absolute Percentage Error (MAPE)**

  $$
    \text{MAPE} 
    = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|.
  $$

  Expresses error as a percentage of true value; undefined if any $y_i=0$.

* **R-squared ($R^2$)**

  $$
    R^2 
    = 1 \;-\; \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}
                 {\sum_{i=1}^n (y_i - \bar{y})^2}.
  $$

  Proportion of variance in $y$ explained by the model (1 is perfect, 0 means no improvement over predicting $\bar{y}$).

* **Adjusted R-squared**

  $$
    \overline{R}^2 
    = 1 \;-\; \bigl(1 - R^2\bigr)\frac{n - 1}{n - p - 1},
  $$

  where $p$ is number of features. Penalizes adding irrelevant features.

* **Explained Variance Score**

  $$
    \text{ExplainedVar} 
    = 1 \;-\; \frac{\operatorname{Var}(y - \hat{y})}{\operatorname{Var}(y)}.
  $$

  Similar to $R^2$; measures how much of the data’s variance is accounted for by the predictions.

* **Huber Loss** (for robustness)

  $$
    L_\delta(y, \hat{y})
    = \begin{cases}
      \tfrac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \le \delta,\\
      \delta\,|y - \hat{y}| - \tfrac{1}{2}\delta^2, & \text{otherwise}.
    \end{cases}
  $$

  Combines MSE and MAE to be less sensitive to outliers when $|y-\hat{y}|$ is large.

---

**Evaluation Metrics (Clustering & Others, for Completeness)**

* **Silhouette Score** (for clustering)

  $$
    s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}},
  $$

  where $a(i)$ = average distance of sample $i$ to other points in its cluster, and $b(i)$ = lowest average distance to points in any other cluster. Ranges from $-1$ to 1.

* **Adjusted Rand Index (ARI)** (for clustering)
  Measures similarity between true labels and cluster assignments, adjusted for chance.

* **Normalized Mutual Information (NMI)** (for clustering)

  $$
    \text{NMI}(U, V) 
    = \frac{2\,I(U;V)}{H(U) + H(V)},
  $$

  where $I(U;V)$ is mutual information between clusterings $U$ and $V$, and $H$ is entropy. Ranges from 0 to 1.

---

**Key Takeaways for Exam Prep**

1. **Classification Metrics**:

   * Always start with a **confusion matrix** (TP, TN, FP, FN).
   * Derive **accuracy**, **precision**, **recall**, **specificity**, and **F1-score** from it.
   * Understand threshold-based curves: **ROC–AUC** and **PR–AUC**.
   * Know **Log Loss** for probability‐based evaluation.
   * Be aware of **MCC** for balanced assessments on skewed datasets.

2. **Regression Metrics**:

   * MSE vs. MAE: MSE penalizes large errors more; MAE is robust to outliers.
   * RMSE is the square root of MSE (same units as target).
   * R² explains variance; Adjusted R² accounts for number of features.
   * MAPE expresses error in percentage terms (watch out for zeros).
   * Huber loss trades off between MSE and MAE when outliers exist.

3. **When to Use Which Metric**:

   * If you care about large outliers, use **MSE**/**RMSE** (heavily penalizes large errors).
   * If you want an interpretable average error, use **MAE**.
   * In classification with imbalanced classes, focus on **precision**, **recall**, **F1-score**, and **ROC–AUC**, not just accuracy.
   * For clustering, use **Silhouette**, **ARI**, or **NMI** since there are no true labels in unsupervised settings.

