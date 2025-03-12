import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

st.set_page_config(page_title="Financial Wellness Recommendation System", layout="wide")

# App title and description
st.title("Financial Wellness Program Recommendation System")
st.markdown("""
This app helps users find financial wellness programs based on their profile and interests.
Upload your data files to get started or use sample data for demonstration.
""")

# Function to load data
@st.cache_data
def load_data(uploaded_files=None):
    """Load data either from uploaded files or use sample data"""
    if uploaded_files:
        # Process uploaded files
        data = {}
        for file in uploaded_files:
            filename = file.name
            if filename.endswith('.csv'):
                data[filename.replace('.csv', '')] = pd.read_csv(file)
        return data
    else:
        # Use sample data (for demonstration)
        return generate_sample_data()

# Function to generate sample data for demonstration
def generate_sample_data():
    # Create sample programs data
    programs = pd.DataFrame({
        'programId': range(1, 11),
        'title': [
            'Debt Management 101', 'Retirement Planning', 'Investment Basics',
            'Budget Mastery', 'Credit Score Improvement', 'Emergency Fund Building',
            'Homebuyer Workshop', 'Tax Planning', 'Estate Planning', 'Financial Independence'
        ],
        'categories': [
            'Debt Management, Financial Literacy', 'Retirement, Investing',
            'Investing, Financial Literacy', 'Budgeting, Financial Literacy',
            'Credit, Financial Literacy', 'Savings, Financial Literacy',
            'Real Estate, Financial Literacy', 'Tax, Financial Literacy',
            'Estate Planning, Financial Literacy', 'FIRE, Financial Literacy'
        ],
        'cost_tier': [
            'Free', 'Premium ($50-$100)', 'Free', 'Basic ($0-$49)',
            'Basic ($0-$49)', 'Free', 'Premium ($50-$100)', 'Premium ($50-$100)',
            'Premium ($50-$100)', 'Premium ($50-$100)'
        ],
        'format': [
            'Video, Interactive', 'Video, Webinar', 'Text, Interactive',
            'Video, Text', 'Interactive, Text', 'Video', 'In-person, Video',
            'Video, Text', 'Video, Webinar', 'Video, Text, Interactive'
        ],
        'instructor_expertise': [
            'Certified Financial Planner', 'Certified Financial Planner, Financial Advisor',
            'Financial Educator', 'Financial Coach', 'Credit Counselor',
            'Financial Coach', 'Real Estate Agent, Financial Advisor',
            'CPA, Tax Expert', 'Estate Attorney', 'Financial Independence Expert'
        ],
        'PCA1': [0.35, 0.42, 0.28, 0.15, 0.22, 0.18, 0.45, 0.38, 0.40, 0.48],
        'PCA2': [0.28, 0.32, 0.22, 0.12, 0.25, 0.15, 0.38, 0.28, 0.35, 0.42]
    })
    
    # Create sample users data
    users = pd.DataFrame({
        'userId': range(1, 101),
        'age': np.random.randint(22, 65, 100),
        'income_bracket': np.random.choice(['Low', 'Medium', 'High'], 100),
        'financial_goals': np.random.choice([
            'Debt reduction, Saving', 'Retirement, Investing', 'Home purchase, Debt reduction',
            'Investing, Financial independence', 'Debt reduction, Credit improvement'
        ], 100),
    })
    
    # Add age bracket
    users['age_bracket'] = users['age'].apply(
        lambda x: 'Young' if x < 33 else ('Middle-aged' if x <= 48 else 'Older')
    )
    
    # Create sample financial profiles
    financial_profiles = pd.DataFrame({
        'userId': range(1, 101),
        'debt_level': np.random.choice(['Low', 'Medium', 'High', 'None'], 100),
        'savings_rate': np.random.choice(['Low', 'Medium', 'High'], 100),
        'investment_experience': np.random.choice(['None', 'Beginner', 'Intermediate', 'Advanced'], 100),
        'financial_stress_level': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    
    # Create sample program tags
    program_tags = pd.DataFrame({
        'programId': np.random.choice(range(1, 11), 100),
        'userId': range(1, 101)
    })
    
    # Create sample ratings
    ratings = pd.DataFrame({
        'userId': np.random.choice(range(1, 101), 200),
        'programId': np.random.choice(range(1, 11), 200),
        'rating': np.random.randint(1, 6, 200)
    }).drop_duplicates(['userId', 'programId']).reset_index(drop=True)
    
    # Merge users and financial profiles
    users_with_finance = users.merge(financial_profiles, on='userId', how='inner')
    users_with_finance = users_with_finance.merge(program_tags[['userId', 'programId']], on='userId', how='inner')
    users_with_finance = users_with_finance.merge(
        programs[['programId', 'title', 'PCA1', 'PCA2']], on='programId', how='inner'
    )
    
    # Create combined features
    users_with_finance['combined_features'] = (
        users_with_finance['income_bracket'] + ', ' + 
        users_with_finance['financial_goals'] + ', ' +
        users_with_finance['age_bracket'] + ', ' +
        users_with_finance['debt_level'] + ', ' +
        users_with_finance['savings_rate'] + ', ' +
        users_with_finance['investment_experience'] + ', ' +
        users_with_finance['financial_stress_level'] + ' ' +
        users_with_finance['PCA1'].astype('str') + ' ' +
        users_with_finance['PCA2'].astype('str')
    )
    
    return {
        'programs': programs,
        'users': users,
        'financial_profiles': financial_profiles,
        'program_tags': program_tags,
        'ratings': ratings,
        'users_with_finance': users_with_finance
    }

# Option to upload data files or use sample data
data_option = st.radio(
    "Choose data source:",
    ["Use sample data", "Upload your own data"]
)

if data_option == "Upload your own data":
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type="csv")
    if uploaded_files:
        data = load_data(uploaded_files)
        st.success(f"Uploaded {len(uploaded_files)} files.")
    else:
        st.info("Please upload files or select 'Use sample data'.")
        data = None
else:
    data = load_data()
    st.success("Using sample data for demonstration.")

# Only proceed if data is available
if data:
    # Preprocess data and create recommendation engine
    @st.cache_data
    def create_recommendation_engine(users_with_finance):
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        program_matrix = vectorizer.fit_transform(users_with_finance['combined_features'])
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(program_matrix, program_matrix)
        
        # Create indices
        indices = pd.Series(range(len(users_with_finance)), index=users_with_finance['title']).to_dict()
        
        return cosine_sim, indices

    # Function to make content-based recommendations
    def make_recommendations(program, N, users_with_finance, cosine_sim, indices):
        if program not in users_with_finance['title'].values:
            return pd.DataFrame({"Error": ["Program Does Not Exist"]})
        
        idx = indices[program]

        sim_score = list(enumerate(cosine_sim[idx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
        sim_score = sim_score[1:]

        program_idx = [x[0] for x in sim_score]
        similarity = [x[1] for x in sim_score]

        recommendations = users_with_finance.iloc[program_idx][['title', 'financial_goals', 'age_bracket']].reset_index(drop=True)
        recommendations['similarity_score'] = np.round(similarity, 2)

        original_program = pd.DataFrame({
            'title': [program], 
            'financial_goals': [users_with_finance.loc[idx, 'financial_goals']], 
            'age_bracket': [users_with_finance.loc[idx, 'age_bracket']], 
            'similarity_score': [1]
        })
        
        recommendations = pd.concat([original_program, recommendations], ignore_index=True)
        recommendations = recommendations.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        return recommendations.head(N)

    # Function to generate predicted ratings
    def content_generate_ratings(program, user, rating_data, cosine_sim, indices, k=20, threshold=0.0):
        if program not in indices.keys():
            return "Program does not exist"
        
        idx = indices[program]
        
        programs_rated = rating_data[rating_data['userId'] == user]
        
        neighbors = []
        
        for _, row in programs_rated.iterrows():
            if row['title'] not in indices:
                continue
                
            rated_idx = indices[row['title']]

            if idx >= cosine_sim.shape[0] or rated_idx >= cosine_sim.shape[1]:
                continue

            similarity = cosine_sim[idx, rated_idx]
            neighbors.append((similarity, row['rating']))
            
        k_neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:k]

        sim_total, weighted_ratings = 0, 0
        for k in k_neighbors:
            if k[1] >= threshold:
                sim_total += k[0]
                weighted_ratings += k[0] * k[1]

        if sim_total == 0:
            return np.mean(rating_data[rating_data['title'] == program]['rating']) if not rating_data[rating_data['title'] == program]['rating'].empty else 3.0

        return weighted_ratings/sim_total

    # Create recommendation engine
    users_with_finance = data['users_with_finance']
    cosine_sim, indices = create_recommendation_engine(users_with_finance)
    
    # Create rating data for predictions
    rating_data = users_with_finance.merge(
        data['ratings'][['userId', 'programId', 'rating']], 
        on='programId', 
        how='inner'
    )[['title', 'userId', 'rating']].drop_duplicates()

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Program Recommendations", "Rating Predictions", "User Profiles"])

    with tab1:
        st.header("Find Similar Programs")
        
        # Select program for recommendations
        program_options = sorted(users_with_finance['title'].unique())
        selected_program = st.selectbox(
            "Select a financial wellness program:",
            program_options
        )
        
        num_recommendations = st.slider(
            "Number of recommendations:", 
            min_value=1, 
            max_value=10, 
            value=5
        )
        
        if st.button("Get Recommendations"):
            with st.spinner("Finding similar programs..."):
                recommendations = make_recommendations(
                    selected_program, 
                    num_recommendations, 
                    users_with_finance, 
                    cosine_sim, 
                    indices
                )
                
                st.dataframe(recommendations)
                
                # Visualize similarity scores
                if len(recommendations) > 1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.barh(
                        recommendations['title'][1:], 
                        recommendations['similarity_score'][1:],
                        color='skyblue'
                    )
                    ax.set_xlabel('Similarity Score')
                    ax.set_title('Program Similarity to ' + selected_program)
                    ax.set_xlim([0, 1])
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width + 0.01, 
                            bar.get_y() + bar.get_height()/2, 
                            f'{width:.2f}', 
                            va='center'
                        )
                    
                    st.pyplot(fig)

    with tab2:
        st.header("Predict Program Ratings")
        
        # User selection
        user_options = sorted(rating_data['userId'].unique())
        selected_user = st.selectbox(
            "Select a user ID:",
            user_options
        )
        
        # Program selection for rating prediction
        program_options_rating = sorted(rating_data['title'].unique())
        selected_program_rating = st.selectbox(
            "Select a program to predict rating:",
            program_options_rating,
            key="program_rating"
        )
        
        if st.button("Predict Rating"):
            with st.spinner("Calculating predicted rating..."):
                # Get actual rating if available
                actual_rating = rating_data[
                    (rating_data['userId'] == selected_user) & 
                    (rating_data['title'] == selected_program_rating)
                ]['rating']
                
                actual_rating_value = "No actual rating available" if actual_rating.empty else actual_rating.values[0]
                
                # Predict rating
                predicted_rating = content_generate_ratings(
                    selected_program_rating, 
                    selected_user, 
                    rating_data,
                    cosine_sim,
                    indices
                )
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Actual Rating", actual_rating_value)
                with col2:
                    st.metric("Predicted Rating", f"{predicted_rating:.2f}")
                
                # Get user's other ratings for comparison
                user_ratings = rating_data[rating_data['userId'] == selected_user].sort_values('rating', ascending=False)
                
                if not user_ratings.empty:
                    st.subheader("User's Other Program Ratings")
                    st.dataframe(user_ratings)
                    
                    # Visualize ratings
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='rating', y='title', data=user_ratings, ax=ax, palette='viridis')
                    ax.set_xlabel('Rating')
                    ax.set_ylabel('Program')
                    ax.set_title(f'User {selected_user} Program Ratings')
                    st.pyplot(fig)

    with tab3:
        st.header("User Profiles Analysis")
        
        # User demographic visualizations
        st.subheader("User Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data['users']['age'], bins=10, kde=True, ax=ax)
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            ax.set_title('Age Distribution')
            st.pyplot(fig)
            
            # Financial goals
            fig, ax = plt.subplots(figsize=(8, 5))
            goals_count = data['users']['financial_goals'].value_counts().reset_index()
            goals_count.columns = ['Goal', 'Count']
            sns.barplot(x='Count', y='Goal', data=goals_count, ax=ax)
            ax.set_title('Financial Goals Distribution')
            st.pyplot(fig)

        with col2:
            # Income distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            income_count = data['users']['income_bracket'].value_counts().reset_index()
            income_count.columns = ['Income', 'Count']
            sns.barplot(x='Count', y='Income', data=income_count, ax=ax)
            ax.set_title('Income Bracket Distribution')
            st.pyplot(fig)
            
            # Age bracket
            fig, ax = plt.subplots(figsize=(8, 5))
            age_count = data['users']['age_bracket'].value_counts().reset_index()
            age_count.columns = ['Age Bracket', 'Count']
            sns.barplot(x='Count', y='Age Bracket', data=age_count, ax=ax)
            ax.set_title('Age Bracket Distribution')
            st.pyplot(fig)
        
        # Program ratings distribution
        st.subheader("Program Ratings Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='rating', data=data['ratings'], ax=ax)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Overall Rating Distribution')
        st.pyplot(fig)
        
        # Program popularity
        st.subheader("Program Popularity")
        
        program_counts = data['ratings']['programId'].value_counts().reset_index()
        program_counts.columns = ['programId', 'count']
        program_counts = program_counts.merge(
            data['programs'][['programId', 'title']], 
            on='programId', 
            how='left'
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='count', y='title', data=program_counts, ax=ax)
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Program')
        ax.set_title('Program Popularity by Rating Count')
        st.pyplot(fig)

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    ### About this App
    This Streamlit app provides a recommendation system for financial wellness programs.
    
    **Features:**
    - Content-based program recommendations
    - Rating predictions based on user profiles
    - Analysis of user demographics and program popularity
    
    Built with Streamlit, Pandas, Scikit-learn, and Matplotlib.
    """)
else:
    st.warning("No data available. Please upload files or use sample data.")