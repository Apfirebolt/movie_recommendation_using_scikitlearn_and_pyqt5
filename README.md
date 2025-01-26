# A GUI application Movie Recommendation App

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