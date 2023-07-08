
# Sentinet: Sentiment Analysis of IMDb Movie Reviews

Sentinet is a web application that performs sentiment analysis on IMDb movie reviews. It leverages natural language processing techniques to determine the sentiment (positive or negative) expressed in a given movie review. The application also provides movie details fetched from the OMDB API, including the movie title, release year, genre, director, actors, and plot. Additionally, it displays a movie poster and retrieves user reviews for a specific movie and visualizes the overall sentiment.





## Demo




## Features

- **Sentiment Analysis**: Utilizes natural language processing and Neural networks to analyze the sentiment (Positive or Negative) of IMDb movie reviews.
- **Text Preprocessing**: Uses regular expressions (re) and the Natural Language Toolkit (NLTK) stopwords to preprocess the movie reviews.
- **Vector Representation**: Utilizes the CountVectorizer from scikit-learn to convert the preprocessed reviews into a vector representation(Method BOW).
- **OMDB API Integration**: Fetches movie details, including title, release year, genre, director, actors, and plot, from the OMDB API.
- **User Reviews**: When the user clicks on "Show Reviews" retrieves a set of sample reviews for the entered movie and displays them for the user when user clicks "Show Reviews".
- **Overall Sentiment**: Visualize the overall sentiment Score of the movie based on the fetched reviews.
- **Analyze Review**: When user enters or paste review in input field then it will show Sentiment Result, Confidence Score, Word Cloud, Top Keywords when user clicks Analyze.
- **Word Cloud**: Generates a word cloud representation of the entered movie review.
- **Top Keywords**: Extracts and displays the top keywords from the entered movie review.
- **Confidence Score**: Calculates the confidence score of the sentiment analysis result.


## Technologies Used

- Streamlit: A Python library used to build the web application interface.
- TensorFlow and Keras: Used for building and loading the Artificial Neural Network (ANN) model.
- NLTK: Used for preprocessing the movie reviews, including removing stopwords and lemmatization.
- scikit-learn: Used for vector representation and analysis of the movie review text.
- WordCloud and Matplotlib: Utilized to generate and display the word cloud visualization.
- Requests: Used for making API calls to the OMDB API.
- IMDbPY: A Python package used to fetch movie details and reviews from the IMDb website.

## Instructions

1. Enter the movie title, release year, and genre in the sidebar of the web application.
2. Click the "Okay" button to set the movie details.
3. The movie details will be displayed in the main area, including the movie poster if available.
4. Click the "Show Reviews" button to fetch and display user reviews for the movie.
5. Click the "Overall Sentiment" button to visualize the overall sentiment based on the fetched movie reviews.
6. Enter or paste a movie review in the text area provided.
7. Click the "Analyze" button to perform sentiment analysis on the entered review.
8. The sentiment analysis result, confidence score, review length, word cloud, and top keywords will be displayed.
## Usage/Examples

**For Users**:
- **Make Informed Movie Choices:** Users can enter a movie title and analyze the sentiment of its reviews to get an understanding of the overall positive or negative sentiment associated with the movie. This helps them make informed decisions when choosing which movies to watch.

- **Explore User Reviews:** Sentinet allows users to fetch and explore user reviews for a specific movie. By reading these reviews, users can gain insights into other viewers' opinions and experiences with the movie.

**For Companies:**

- **Market Research and Consumer Insights:** Sentinet can be used by companies in the film industry for market research and to gain consumer insights. By analyzing the sentiment of IMDb movie reviews, companies can understand audience reactions to movies and make data-driven decisions for marketing campaigns, movie promotions, and content creation.

- **Movie Recommendation Systems:** Sentinet's sentiment analysis capability can be integrated into movie recommendation systems. By analyzing the sentiment of user reviews, companies can improve the accuracy of their recommendation algorithms and provide personalized movie recommendations to users based on their sentiment preferences.

- **Brand Monitoring:** Sentinet can be used by production companies and studios to monitor the sentiment of reviews and social media discussions surrounding their movies. This helps them gauge audience reactions, identify potential issues or criticisms, and take appropriate actions for reputation management and brand enhancement.


## License




This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the LICENSE file for more information.
## Acknowledgements

- The creators and contributors of the open-source libraries used in this project.
- The OMDB API for providing access to movie details.
- The IMDbPY project for fetching movie reviews and details from the IMDb website.



