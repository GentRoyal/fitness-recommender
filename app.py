import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data (Modify this to load actual data from CSVs)
@st.cache_data
def load_data():
    users_with_finance = pd.read_csv("users_with_finance.csv")  # Replace with actual path
    return users_with_finance

def preprocess_data(users_with_finance):
    vectorizer = TfidfVectorizer()
    program_matrix = vectorizer.fit_transform(users_with_finance['combined_features'])
    cosine_sim = cosine_similarity(program_matrix, program_matrix)
    return users_with_finance, cosine_sim

def make_recommendations(program, N, users_with_finance, cosine_sim):
    if program not in users_with_finance['title'].values:
        return 'Program Does Not Exist'
    
    idx = users_with_finance[users_with_finance['title'] == program].index[0]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:N+1]
    
    program_idx = [x[0] for x in sim_score]
    similarity = [x[1] for x in sim_score]
    
    recommendations = users_with_finance.iloc[program_idx][['title', 'financial_goals', 'age_bracket']].reset_index(drop=True)
    recommendations['similarity_score'] = np.round(similarity, 2)
    
    return recommendations

# Streamlit UI
st.title("Program Recommender System")
users_with_finance = load_data()
users_with_finance, cosine_sim = preprocess_data(users_with_finance)

selected_program = st.selectbox("Select a Program", users_with_finance['title'].unique())
N = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    recommendations = make_recommendations(selected_program, N, users_with_finance, cosine_sim)
    st.write("### Recommended Programs:")
    st.dataframe(recommendations)
