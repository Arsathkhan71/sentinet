import streamlit as st
# from transformers import pipeline
from tensorflow import keras
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from imdb import IMDb
import gdown
import h5py
nltk.download("stopwords")

# Load the sentiment analysis model
model = keras.models.load_model("modelANN.h5")

# Load CountVectorizer vocabulary
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

# header image
st.image("sentiment-analysis-2.png", use_column_width=True)

# Set Streamlit app title and sidebar information
st.title("IMDB Movie Review Sentiment Analysis")
st.sidebar.title("Movie Details")

# Movie details inputs
title = st.sidebar.text_input("Movie Title", "")
release_year = st.sidebar.number_input("Release Year", min_value=1800, max_value=2100, step=1)
genre = st.sidebar.text_input("Genre", "")

# Preprocess the input review
def preprocess_text(text):
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', text)
    
    # Convert to lowercase
    clean_text = clean_text.lower()
    
    # Remove URLs
    clean_text = re.sub(r'http\S+', '', clean_text)
    
    # Remove mentions
    clean_text = re.sub(r'@(\w+)', '', clean_text)
    
    # Remove hashtags
    clean_text = re.sub(r'#(\w+)', '', clean_text)
    
    # Remove special characters and punctuation
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    
    # Remove numeric characters
    clean_text = re.sub(r'\d+', '', clean_text)
    
    # Remove extra whitespaces
    clean_text = ' '.join(clean_text.split())
    
    # Lemmatization
#     lemmatizer = WordNetLemmatizer()
#     clean_text = ' '.join([lemmatizer.lemmatize(word) for word in clean_text.split()])
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join([word for word in clean_text.split() if word not in stop_words])
    
    return clean_text

# Function to fetch movie details from OMDB API
def get_movie_details(movie_name):
    # API endpoint
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey=d8c87ec5"
    
    # Send GET request to the API
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return None
    
# Okay button
if st.sidebar.button("Okay"):
    st.sidebar.markdown("Movie details have been set.")
    
# Fetch movie details if movie name is entered
if title:
    movie_details = get_movie_details(title)
    if movie_details:
        st.header("Movie Details")
        st.markdown(f"Title: {movie_details['Title']}")
        st.markdown(f"Year: {movie_details['Year']}")
        st.markdown(f"Genre: {movie_details['Genre']}")
        st.markdown(f"Director: {movie_details['Director']}")
        st.markdown(f"Actors: {movie_details['Actors']}")
        st.markdown(f"Plot: {movie_details['Plot']}")
        poster_url = movie_details['Poster']
        if poster_url != 'N/A':
            poster = Image.open(requests.get(poster_url, stream=True).raw)
            # poster.thumbnail((100, 200))
            st.image(poster, caption='Movie Poster', use_column_width=True)
        st.sidebar.markdown("---")
    else:
        st.sidebar.warning("Failed to retrieve movie details")
        

    # Show Reviews button
    if st.button("Show Reviews"):
        st.markdown("Fetching reviews...")

        # Create an instance of IMDb class
        ia = IMDb()

        # Search for the movie by title
        movies = ia.search_movie(title)
        found_movie = None

        # # Find the exact matching movie
        # for movie in movies:
        #     if movie['title'].lower() == title.lower():
        #         found_movie = movie
        #         break

        if movies:
            movie_id = movies[0].movieID
            movie = ia.get_movie(movie_id)

            # Get the user reviews
            ia.update(movie, 'reviews')

            st.header("User Reviews")

            # Display the first 5 reviews
            # review_count = min(5, len(movie['reviews']))
            if 'reviews' in movie.keys():
                review_text = ""
                for review in movie['reviews']:
                    review_text += f"**Author**: {review['author']}\n"
                    review_text += f"**Rating**: {review['rating']}\n"
                    review_text += f"**Content**: {review['content']}\n"
                    review_text += "---\n"
                st.text_area("Reviews", value=review_text, height=200) 
                    

            else:
                st.markdown("No user reviews available for this movie.")
        else:
            st.warning("Movie not found")
    
    if st.button("Overall Sentiment"):
        st.markdown("Fetching Overall Sentiment Score...")
        
        # Create an instance of IMDb class
        ia = IMDb()
        
        # Search for the movie by title
        movies = ia.search_movie(title)
        if movies:
            movie_id = movies[0].movieID
            movie = ia.get_movie(movie_id)

            # Get the user reviews
            ia.update(movie, 'reviews')

            # print("User Reviews")

            # Initialize counters for positive and negative reviews
            positive_count = 0
            negative_count = 0

            # Display the fetched reviews
            if 'reviews' in movie.keys():
                for review in movie['reviews']:
        #             print(f"**Author**: {review['author']}")
        #             print(f"**Rating**: {review['rating']}")
        #             print(f"**Content**: {review['content']}\n")

                    # Perform sentiment analysis on the review content
                    # Add your sentiment analysis method here
                    def analyze_sentiment(review):
                        # review_pr = preprocess_text(review)
                        review_cv = cv.transform([review])
                        sentiment = model.predict(review_cv)
                        return sentiment

                    # Assume the sentiment analysis method returns a label ('positive' or 'negative')
                    sentiment_label = analyze_sentiment(review['content'])
                    # print(sentiment_label)

                    # Update the counters based on the sentiment label
                    if sentiment_label >= 0.5:
                        positive_count += 1
                    else:
                        negative_count += 1
            else:
                print("No user reviews available for this movie.")

            # Calculate the percentage of positive and negative reviews
            total_reviews = positive_count + negative_count
            if total_reviews > 0:
                positive_percentage = (positive_count / total_reviews) * 100
                negative_percentage = (negative_count / total_reviews) * 100
                print(f"\nOverall Sentiment:")
                print(f"Positive: {positive_percentage:.2f}%")
                print(f"Negative: {negative_percentage:.2f}%")
                
        #          Create a bar chart to visualize the overall sentiment
                labels = ['Positive', 'Negative']
                sentiment_percentages = [positive_percentage, negative_percentage]
                colors = ['#34A853', '#EA4335']  # Green for positive, Red for negative
                
                # plt.figure(figsize=(5, 4))
                # plt.bar(labels, sentiment_percentages, color=colors)
                # plt.xlabel('Sentiment')
                # plt.ylabel('Percentage')
                # plt.title('Overall Sentiment of Reviews')
                # plt.axis("off")
                # plt.ylim([0, 100])  # Set the y-axis limit from 0 to 100
                # plt.show()
                # st.pyplot(plt)
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.barh(labels, sentiment_percentages, color=colors)
                ax.set_xlabel('Percentage')
                # ax.set_title('Overall Sentiment of Reviews')
                ax.set_xlim([0, 100])  # Set the x-axis limit from 0 to 100

                # Display the bar values inside the bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", ha='left', va='center')
                 
                # Turn off the X-axis
                ax.axes.get_xaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # Add a legend with the percentage values
                legend_labels = [f"{label}: {percentage:.2f}%" for label, percentage in zip(labels, sentiment_percentages)]
                ax.legend(bars, legend_labels)
                
                # Adjust the padding between the plot elements
                plt.tight_layout()

                # Remove the default padding around the plot
                plt.margins(0)

                st.pyplot(fig)
                
                
            else:
                print("No sentiment analysis performed due to lack of reviews.")
        else:
            print("Failed to retrieve movie details")
                    
            
        
# Input text area for the movie review
review_text = st.text_area("Enter the movie review:", "")
review_text_pr = preprocess_text(review_text)
review_text_cv = cv.transform([review_text_pr])



# Analyze button
if st.button("Analyze"):

        # Calculate review length
        review_length = len(review_text)
        # Display review length
        st.markdown(f"Review Length: {review_length} characters")
        

        # Perform sentiment analysis on the input review
        sentiment = model.predict(review_text_cv)
        

        # Display the sentiment result
        st.header("Sentiment Analysis Result")
        if sentiment > 0.5:
            st.subheader("Positive😊")
        else:
            st.subheader("Negative😢")
        # st.markdown(f"Sentiment: **{sentiment_label}**")
        # st.markdown(f"Confidence Score: {sentiment_score:.4f}")
        
        
        ## Confidence Score
        prob = model.predict(review_text_cv)[0][0]*100
        st.markdown(f"Confidence Score: {prob:.2f}"+'%')


        # Display movie details
        st.sidebar.subheader("Movie Details")
        st.sidebar.markdown(f"**Title:** {title}")
        st.sidebar.markdown(f"**Release Year:** {release_year}")
        st.sidebar.markdown(f"**Genre:** {genre}")
        
        st.markdown("_____________________________________________________________________________________________________")
        ## Word Cloud
        wordcloud = WordCloud().generate(preprocess_text(review_text))
        st.header("Word Cloud")
        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(plt)
        
        st.markdown("_____________________________________________________________________________________________________")

        ## Top Keywords
        # Create TF-IDF vectorizer
        st.header("Top Keywords")
        vectorizer = TfidfVectorizer()

        # Fit and transform the input review
        review_vector = vectorizer.fit_transform([preprocess_text(review_text)])

        # Get the feature names (keywords)
        feature_names = vectorizer.get_feature_names_out()

        # Get the TF-IDF scores for the keywords
        tfidf_scores = review_vector.toarray()[0]

        # Get the indices of the top keywords based on TF-IDF scores
        top_indices = tfidf_scores.argsort()[-50:][::-1]

        # Get the top keywords
        top_keywords = [feature_names[i] for i in top_indices]

        # Display the top keywords
        keyword_list = [keyword for keyword in top_keywords]
        keyword_str = ", ".join(keyword_list)
        st.markdown(keyword_str, unsafe_allow_html=True)
        
        st.markdown("_____________________________________________________________________________________________________")

        # # Display movie details if entered
        # if title or (release_year != 1800) or genre:
        #     st.header("Movie Details")
        #     if title:
        #         st.markdown("Title: " + title)
        #     if release_year != 1800:
        #         st.markdown("Release Year: " + str(release_year))
        #     if genre:
        #         st.markdown("Genre: " + genre)
        
        # Display movie details
        
                



# Input text area for the movie review
# movie_name = st.sidebar.text_input("Movie Name", "")
       

    
    # # Fetch movie details from the API
    # movie_details = get_movie_details(title)
    
    # # Display movie details if retrieved successfully
    # if movie_details:
    #     st.header("Movie Details")
    #     st.markdown(f"Title: {movie_details['Title']}")
    #     st.markdown(f"Year: {movie_details['Year']}")
    #     st.markdown(f"Genre: {movie_details['Genre']}")
    #     # Add more movie details as desired
    # else:
    #     st.sidebar.warning("Failed to retrieve movie details")

# Clear button
if st.sidebar.button("Clear"):
    # Clear the movie details inputs
    title = ""
    release_year = 1800
    genre = ""
    # Clear the sidebar display
    st.sidebar.empty()
                

# Instructions and description
st.subheader("Instructions")
st.markdown("1. Enter or paste the movie review in the input area.")
st.markdown("2. Provide optional Valid movie details in the sidebar(Check once in Browser).")
st.markdown("3. Click the **Analyze** button to perform sentiment analysis.")
st.markdown("4. The sentiment analysis result will be displayed below.")

# Add more elements and visualizations as desired
# such as word cloud, sentiment trend chart, etc.
