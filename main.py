import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
)
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtCore import Qt, QTimer


class SimpleTable(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Movie Recommendation System")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table)


class PaginatedTable(SimpleTable):

    def __init__(self, data, rows_per_page=20):
        self.data = data
        self.rows_per_page = rows_per_page
        self.current_page = 0
        self.model = None
        self.tfidf_matrix = None
        super().__init__()


    def train_model_overview(self):
        # Create a TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the overview column
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['overview'].fillna(''))

        # Compute the cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return cosine_sim
    
    
    def get_recommendations_cosine_sim(self,title, cosine_sim):

        # Return the 5 top matches
        # Get the index of the movie that matches the title
        idx = self.data[self.data['title'] == title].index[0]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 5 most similar movies
        sim_scores = sim_scores[1:6]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar movies
        return self.data['title'].iloc[movie_indices]
    

    def train_model(self):
        # Create a TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the keywords column
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['keywords'].fillna(''))

        # Compute the cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return cosine_sim


    def train_knn_model(self):
        # Create a TF-IDF Vectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the overview column
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['overview'].fillna(''))

        # Train the KNN model
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(self.tfidf_matrix)

        return knn, self.tfidf_matrix

    def get_knn_recommendations(self, title, knn, tfidf_matrix):
        try:
            # Get the index of the movie that matches the title
            idx = self.data[self.data['title'] == title].index[0]

            # Get the pairwise similarity scores of all movies with that movie
            distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=6)

            # Get the movie indices
            movie_indices = indices.flatten()[1:]

            # Return the top 5 most similar movies
            return self.data['title'].iloc[movie_indices]
        
        except IndexError:
            self.error_window = QWidget()
            self.error_window.setWindowTitle("Error")
            self.error_window.setGeometry(200, 200, 300, 100)

            layout = QVBoxLayout()
            self.error_window.setLayout(layout)

            error_label = QLabel(f"Movie '{title}' not found.")
            layout.addWidget(error_label)

            self.error_window.show()
            return []
    
    def get_recommended_movies(self):
        movie_title = self.movie_input.text()
        recommendations = self.get_knn_recommendations(movie_title, self.model, self.tfidf_matrix)
        
        if len(recommendations):
            self.recommendations_window = QWidget()
            self.recommendations_window.setWindowTitle(f"Recommendations for '{movie_title}'")
            self.recommendations_window.setGeometry(200, 200, 400, 300)

            layout = QVBoxLayout()
            self.recommendations_window.setLayout(layout)

            for movie in recommendations:
                movie_label = QLabel(movie)
                movie_label.setStyleSheet("padding: 5px; font-size: 14px;")
                layout.addWidget(movie_label)
            self.recommendations_window.show()

    
    def get_cosine_recommendations(self):
        movie_title = self.movie_input.text()
        cosine_sim = self.train_model_overview()
        recommendations = self.get_recommendations_cosine_sim(movie_title, cosine_sim)
        
        if len(recommendations):
            self.recommendations_window = QWidget()
            self.recommendations_window.setWindowTitle(f"Recommendations for '{movie_title}'")
            self.recommendations_window.setGeometry(200, 200, 400, 300)

            layout = QVBoxLayout()
            self.recommendations_window.setLayout(layout)

            for movie in recommendations:
                movie_label = QLabel(movie)
                movie_label.setStyleSheet("padding: 5px; font-size: 14px;")
                layout.addWidget(movie_label)
            self.recommendations_window.show()
        

    def reload_data(self):
        self.data = pd.read_csv("movies.csv")
        self.clean_data()

    def clean_data(self):
        # remove columns like keywords, homepage, id, revenue, status, tagline, vote_count, vote_average, genres, budget
        columns_to_remove = [
            "keywords", "homepage", "id", "revenue", "status", "tagline", "original_language", "production_companies", "production_countries",
            "genres", "budget", "spoken_languages"
        ]
        self.data.drop(columns=columns_to_remove, inplace=True)
        self.model, self.tfidf_matrix = self.train_knn_model()
        self.update_table()

    def search_data(self):
        search_term = self.search_input.text()
        if search_term:
            self.data = self.data[self.data["title"].str.contains(search_term, case=False)]
        else:
            self.data = pd.read_csv("movies.csv")
            self.clean_data()
    
        self.current_page = 0
        self.update_table()

    def initUI(self):
        super().initUI()
        self.clean_data()

        # Add search bar
        search_layout = QHBoxLayout()
        self.search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term")
        self.search_input.textChanged.connect(self.search_data)
        search_layout.addWidget(self.search_label)
        search_layout.addWidget(self.search_input)

        # Add navigation buttons
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.info_button = QPushButton("Info")
        self.reload_button = QPushButton("Reload Data")
        self.reload_button.setStyleSheet(
            "padding: 10px; background-color: #3498db; color: white;"
        )
        self.reload_button.clicked.connect(self.reload_data)
        button_layout.addWidget(self.reload_button)

        # Apply styles
        self.prev_button.setStyleSheet(
            "padding: 10px; background-color: #3498db; color: white;"
        )
        self.next_button.setStyleSheet(
            "padding: 10px; background-color: #3498db; color: white;"
        )
        self.info_button.setStyleSheet(
            "padding: 10px; background-color: #3498db; color: white;"
        )
        self.prev_button.clicked.connect(self.previous_page)
        self.next_button.clicked.connect(self.next_page)
        self.info_button.clicked.connect(self.show_info)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.info_button)
        button_layout.addWidget(self.next_button)

        # A new HBoxLayout for prediction, it should take sex and age as input and have predict button
        prediction_layout = QHBoxLayout()

        self.movie_label = QLabel("Enter Movie Name:")
        self.movie_input = QLineEdit()
        self.recommend_button = QPushButton("Recommend Movies")

        prediction_layout.addWidget(self.movie_label)
        prediction_layout.addWidget(self.movie_input)
        prediction_layout.addWidget(self.recommend_button)

        self.layout().addLayout(search_layout)
        self.layout().addLayout(prediction_layout)
        self.layout().addLayout(button_layout)

        # self.recommend_button.clicked.connect(self.get_recommended_movies)
        self.recommend_button.clicked.connect(self.get_cosine_recommendations)
       

    def update_table(self):
        start_row = self.current_page * self.rows_per_page
        end_row = start_row + self.rows_per_page
        page_data = self.data.iloc[start_row:end_row]

        self.table.setRowCount(len(page_data))
        self.table.setColumnCount(len(page_data.columns))
        # Set the headers by capitalizing column names and converting _ to space
        page_data.columns = [col.replace('_', ' ').capitalize() for col in page_data.columns]
        self.table.setHorizontalHeaderLabels(page_data.columns)

        for i in range(len(page_data)):
            for j in range(len(page_data.columns)):
                item = QTableWidgetItem(str(page_data.iat[i, j]))
                if j == 1:  # If it's the second column
                    item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    item.setData(Qt.UserRole, page_data.iat[i, j])
                self.table.setItem(i, j, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Adjust column sizes
        self.table.setColumnWidth(1, 200)  # Increase the size of the second column
        self.table.setColumnWidth(len(page_data.columns) - 2, 50)  # Decrease the size of the second last column
        self.table.setColumnWidth(len(page_data.columns) - 1, 50)  # Decrease the size of the last column

        self.table.cellClicked.connect(self.cell_was_clicked)

    def cell_was_clicked(self, row, column):
        if column == 1:  # If it's the second column
            cell_content = self.table.item(row, column).data(Qt.UserRole)
            self.show_cell_content(cell_content)

    def show_cell_content(self, content):
        self.content_window = QWidget()
        self.content_window.setWindowTitle("Movie Overview")
        self.content_window.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()
        self.content_window.setLayout(layout)

        content_label = QLabel(content)
        content_label.setWordWrap(True)
        layout.addWidget(content_label)

        self.content_window.show()

    def next_page(self):
        if (self.current_page + 1) * self.rows_per_page < len(self.data):
            self.current_page += 1
            self.update_table()

    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_table()

    def show_info(self):
        info = self.data.describe().to_string()

        self.info_window = QWidget()
        self.info_window.setWindowTitle("Data Info")
        self.info_window.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()
        self.info_window.setLayout(layout)

        info_label = QLabel(info)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.info_window.show()


if __name__ == "__main__":
    data = pd.read_csv("movies.csv")
    app = QApplication(sys.argv)
    window = PaginatedTable(data)
    window.show()
    sys.exit(app.exec_())
