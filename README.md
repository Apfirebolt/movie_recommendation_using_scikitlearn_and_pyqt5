# A GUI application Movie Recommendation App

[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen)](https://en.wikipedia.org/wiki/Machine_learning)
[![NumPy](https://img.shields.io/badge/NumPy-1.21.2-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3.3-yellow)](https://pandas.pydata.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Overview

This application provides movie recommendations based on user preferences. It uses machine learning algorithms to analyze user data and suggest movies that the user is likely to enjoy.

## Features

- User-friendly GUI
- Personalized movie recommendations
- Search functionality
- Detailed movie information

## Algorithms used

### KNN 

K-Nearest Neighbors (KNN) is a simple, yet powerful machine learning algorithm used for classification and regression tasks. In this application, we use the KNN algorithm from the `scikit-learn` library to provide movie recommendations.

### How KNN Works

KNN works by finding the `k` nearest data points (neighbors) to a given input and making predictions based on the majority class (for classification) or the average value (for regression) of these neighbors.

### Implementation in scikit-learn

To implement KNN in our application, we use the `KNeighborsClassifier` from `scikit-learn`. Here is a brief overview of the steps involved:

1. **Import the necessary libraries:**
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    ```

2. **Prepare the data:**
    - Split the data into features and labels.
    - Normalize the data if necessary.

3. **Initialize the KNN classifier:**
    ```python
    knn = KNeighborsClassifier(n_neighbors=5)
    ```

4. **Train the classifier:**
    ```python
    knn.fit(X_train, y_train)
    ```

5. **Make predictions:**
    ```python
    predictions = knn.predict(X_test)
    ```

By using KNN, our application can effectively recommend movies based on the preferences of similar users.


### Cosine Similarity

Cosine Similarity is a metric used to measure how similar two data points are, irrespective of their size. It is often used in text analysis and recommendation systems to find similarities between items.

### How Cosine Similarity Works

Cosine Similarity calculates the cosine of the angle between two vectors in a multi-dimensional space. The cosine value ranges from -1 to 1, where 1 indicates that the vectors are identical, 0 indicates orthogonality (no similarity), and -1 indicates complete dissimilarity.

### Implementation in scikit-learn

To implement Cosine Similarity in our application, we use the `cosine_similarity` function from `scikit-learn`. Here is a brief overview of the steps involved:

1. **Import the necessary library:**
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    ```

2. **Prepare the data:**
    - Convert the data into a suitable format, such as a term-document matrix for text data.

3. **Calculate Cosine Similarity:**
    ```python
    similarity_matrix = cosine_similarity(data_matrix)
    ```

4. **Use the similarity matrix:**
    - The similarity matrix can be used to find the most similar items to a given item, which can then be recommended to the user.

By using Cosine Similarity, our application can effectively recommend movies that are similar to the ones the user has liked in the past.
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/apfirebolt/movie_recommendation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd movie_recommendation
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    python main.py
    ```
2. Interact with the GUI to get movie recommendations.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.