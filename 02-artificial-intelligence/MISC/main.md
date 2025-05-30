### 1. History

### Evaluation Metrics

### 2. Intro to ML ✅
    - Basics, 
    - Classification, 
    - Regression, 
    - Gradient Descent, 
    - Classifiers, 
    - KNN, 
    - SVM, 
    - Linear Regression, 
    - Overfitting, 
    - Regularization
### 3. Neural Networks ✅
    - ANN, CNN, Activation Functions, Parameters
### 4. Agents ✨
    - What, Types, Code: Simple Reflex, Goal Based, Utility; 
    - Agent Environments, Properties of Environment
    - Architecture, Rationality, Autonomy, Single and Multi-agent

1. Simple Reflex
- Act only on the basis of current perception
- Ignores the rest of percept history
- Based on simple if-then rules
- Environment should be fully observable
- Ex: Chess, tempearture sensor

2. Model Based Reflex Agent
- Partially observable environment
- Stores the percept history
- Based on multiple if then rules: if temp> 45, if room is not empty
- Ex: Self driving car

3. Goal Based Agent (Supervised Learning becuase you know the goal/target)
- Expansion of Model based Reflex Agent
- Desireable situation | Goal
- searching and planning
- Ex: delievery agents

4. Utility Based Agent
- Focus on utility not goal
- Utility function: helps us determine the state
- Deals with happy and unhappy state


### 5. Uninformed Search ✨ No heuristics used. Just explore blindly.
search strategy that uses no problem specific knowledge

### 6. Informed Searches ✨ Heuristics = educated guesses (e.g., "How far am I from the goal?")
search strategy that uses problem-specific knowledge to find solutions more efficiently

Greedy Best first search | Heuristic

### A*: search algorithm that expands node with lowest value of g(n) + h(n) 
g(n) = cost to reach node 
h(n) = estimated cost to goal

Ex: pathfinding in maps, robot path planning, 


Adversial Search

Minimax


Search Problems: Agent, state, initial state, actions, goal test, path cost function
Solution: a sequence of actions that leads from the initial state to a goal state
Optimal Solution: a solution that has the lowest path cost among all solutions


| Uninformed Searches | Informed Searches |
|---------------------|-------------------|
|Search without information|Search with information|
|No knowledge| Use knowledge(goal) to find steps towards solution|
|Time consuming|Quick Solution|
|More Complexity|Less complexity|
|Ex: DFS, BFS, etc| A*, Heuristic DFS, Best first search|
| Always optimal solution|No gaurantee|


### BFS 
- Uniformed, blind, or brute force search technique
- FIFO (Queue)
- Shallowest Node

social network, shortest path

### DFS
File system, puzzle

```py
# BFS in pseudocode
queue = [start]
visited = set()

while queue:
    node = queue.pop(0)
    if node not in visited:
        visit(node)
        visited.add(node)
        queue.extend(neighbors of node)

# DFS in pseudocode
stack = [start]
visited = set()

while stack:
    node = stack.pop()
    if node not in visited:
        visit(node)
        visited.add(node)
        stack.extend(neighbors of node)


```

Heuristic: when you want to conver the NP problem to P (Polynomial)




### 7. Dataset ✨✅
    - Types, Issues, optimization techniques, Solutions 



8. [ML Architectures](https://thefaheemkhan.medium.com/list-of-machine-learning-archite-94122f7a17ff) ✅
    - ANN, CNN, RNN (GRU), Transformers, etc.
9. Pre-trained Models Usage ✨ ✅






### 7. Datasets

### Easy Level Answers

1. **Dataset**: A collection of data points or samples used for training or testing machine learning models.

2. **Types of datasets**: Structured, unstructured, and semi-structured.

3. **Difference**:

* Structured data: organized in rows and columns (e.g., Excel spreadsheets).
* Unstructured data: no fixed format (e.g., images, text documents).

4. **Common issues**: Missing data, noisy data, outliers.

5. **Importance of cleaning**: It improves model accuracy and prevents garbage-in-garbage-out by removing errors, inconsistencies, and irrelevant data.

---

### Moderate Level Answers

6. **Missing data**: Data entries with absent values.
   Handling techniques: Imputation (mean/mode filling), deletion of missing records.

7. **Imbalanced dataset**: One class significantly outnumbers others, causing bias in model prediction. Balanced datasets have roughly equal class representation.

8. **Normalization**: Rescaling features to a common scale to improve convergence and performance. Example: Min-Max scaling to \[0,1].

9. **Optimization technique**: Data augmentation — artificially increasing dataset size and diversity to reduce overfitting.

10. **Data augmentation**: Creating modified versions of data (rotations, flips) to improve model robustness and reduce overfitting in image classification.

---

### Advanced Level Answers

11. **Outliers**: Extreme values that can skew model training and results. Detection method: Z-score or IQR (Interquartile Range).

12. **Feature engineering**: Creating or transforming features to improve model performance. Examples: one-hot encoding categorical variables, extracting date parts like day/month.

13. **Addressing imbalance**: Use oversampling (SMOTE), undersampling, and adjust class weights in the loss function.

14. **Overfitting**: Model too complex, learns noise; underfitting: model too simple, misses patterns. Data optimization (augmentation, cleaning) helps generalize model.

15. **Dimensionality reduction**: Reduces feature number, simplifies model. PCA is linear and preserves variance; t-SNE is nonlinear, used for visualization preserving local structure.

---

### Hard Level Answers

16. **Preprocessing pipeline**:

* Handle missing values (imputation).
* Encode categorical variables (one-hot or label encoding).
* Detect/remove outliers (IQR).
* Normalize numerical features (Min-Max or StandardScaler).
* Feature selection to reduce dimensionality (PCA or domain knowledge).

17. **Comprehensive solution**:

* Clean missing/noisy data via imputation and outlier removal.
* Feature selection to remove irrelevant or redundant features.
* Data augmentation if applicable.
* Cross-validation to ensure generalization.

18. **Time-series challenges**: irregular intervals and missing data disrupt temporal continuity.
    **Solution**: Resample data to fixed intervals with interpolation, fill missing values (forward/backward fill), then apply smoothing or feature extraction before modeling.

19. **Synthetic dataset generation**: Use generative models (GANs, variational autoencoders) to mimic real data distribution, combine with domain constraints, validate synthetic data quality, and use regularization during training to avoid overfitting.

20. **Trade-offs**:

* Heavy dimensionality reduction may lose interpretability and important signals.
* Excessive augmentation can introduce unrealistic samples causing bias.
* Optimization may improve accuracy but hurt fairness if minority classes are distorted.
* Example: PCA reduces features but makes model explanations harder; aggressive augmentation may bias model towards artificial features.


---

1. Using Hugging Face Transformers (Local)

```py
pip install transformers torch

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using pre-trained models!")
print(result)

##########
from transformers import pipeline

# Load a pre-trained model for sentiment classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I absolutely love this new phone!")
print(result)

```

2. Using OpenAI GPT (Cloud/API)

```py
pip install openai

import openai
openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Classify text as Positive or Negative."},
        {"role": "user", "content": "I really hate the new update."}
    ]
)

print(response["choices"][0]["message"]["content"])
```

3. Pre Trained CNN
```py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = VGG16(weights='imagenet')

img = load_img('your_image.jpg', target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds, top=1)[0])
```

