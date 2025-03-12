import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data (Modify this to load actual data from CSVs)
@st.cache_data
def load_data():
    users_with_finance = pd.read_csv("users_with_finance.csv")  # Replace with actual path
    rating_data = pd.read_csv("rating_data.csv")  # Replace with actual path
    return users_with_finance, rating_data

def preprocess_data(users_with_finance):
    vectorizer = TfidfVectorizer()
    program_matrix = vectorizer.fit_transform(users_with_finance['combined_features'])
    cosine_sim = cosine_similarity(program_matrix, program_matrix)
    return users_with_finance, cosine_sim

def make_recommendations(program, N, users_with_finance, cosine_sim, threshold=0.0):
    if program not in users_with_finance['title'].values:
        return 'Program Does Not Exist'
    
    idx = users_with_finance[users_with_finance['title'] == program].index[0]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = [x for x in sim_score if x[1] >= threshold][1:N+1]
    
    program_idx = [x[0] for x in sim_score]
    similarity = [x[1] for x in sim_score]
    
    recommendations = users_with_finance.iloc[program_idx][['title', 'financial_goals', 'age_bracket']].reset_index(drop=True)
    recommendations['similarity_score'] = np.round(similarity, 2)
    
    return recommendations

def content_generate_ratings(program, user, rating_data, cosine_sim, k=20, threshold=0.0):
    indices = pd.Series(range(len(users_with_finance)), index=users_with_finance['title']).to_dict()
    if program not in indices.keys():
        return 'Program does not exist'
    idx = indices[program]
    
    programs_rated = rating_data[rating_data['userId'] == user]
    neighbors = []
    
    for _, row in programs_rated.iterrows():
        rated_idx = indices[row['title']]
        similarity = cosine_sim[idx, rated_idx]
        neighbors.append((similarity, row['rating']))
    
    k_neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:k]
    sim_total, weighted_ratings = 0, 0
    for k in k_neighbors:
        if k[1] >= threshold:
            sim_total += k[0]
            weighted_ratings += k[0] * k[1]
    
    if sim_total == 0:
        return np.mean(rating_data[rating_data['title'] == program]['rating'])
    
    return weighted_ratings / sim_total

# Streamlit UI
st.title("Program Recommender System")
users_with_finance, rating_data = load_data()
users_with_finance, cosine_sim = preprocess_data(users_with_finance)

selected_program = st.selectbox("Select a Program", users_with_finance['title'].unique())
N = st.slider("Number of Recommendations", 1, 10, 5)
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.0)
user_id = st.number_input("Enter User ID for Predicted Rating", min_value=1, step=1)

if st.button("Get Recommendations"):
    recommendations = make_recommendations(selected_program, N, users_with_finance, cosine_sim, threshold)
    st.write("### Recommended Programs:")
    st.dataframe(recommendations)

if st.button("Predict Rating"):
    predicted_rating = content_generate_ratings(selected_program, user_id, rating_data, cosine_sim)
    st.write(f"### Predicted Rating for {selected_program}: {predicted_rating:.2f}")
