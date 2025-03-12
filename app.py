import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# App configuration 
st.set_page_config(layout="wide", page_title="Program Recommender System")

# Load data with caching
@st.cache_data
def load_data():
    users_with_finance = pd.read_csv("users_with_finance.csv")  # Replace with actual path
    rating_data = pd.read_csv("rating_data.csv")  # Replace with actual path
    return users_with_finance, rating_data

def preprocess_data(users_with_finance):
    vectorizer = TfidfVectorizer()
    program_matrix = vectorizer.fit_transform(users_with_finance['combined_features'])
    cosine_sim = cosine_similarity(program_matrix, program_matrix)
    return users_with_finance, cosine_sim, vectorizer

def make_recommendations(program, N, users_with_finance, cosine_sim, threshold=0.0, filter_criteria=None):
    if program not in users_with_finance['title'].values:
        return 'Program Does Not Exist'
    
    idx = users_with_finance[users_with_finance['title'] == program].index[0]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = [x for x in sim_score if x[1] >= threshold][1:N+1]
    
    program_idx = [x[0] for x in sim_score]
    similarity = [x[1] for x in sim_score]
    
    recommendations = users_with_finance.iloc[program_idx].reset_index(drop=True)
    recommendations['similarity_score'] = np.round(similarity, 2)
    
    # Apply filters if provided
    if filter_criteria and 'age_bracket' in filter_criteria and filter_criteria['age_bracket']:
        recommendations = recommendations[recommendations['age_bracket'].isin(filter_criteria['age_bracket'])]
    if filter_criteria and 'financial_goals' in filter_criteria and filter_criteria['financial_goals']:
        recommendations = recommendations[recommendations['financial_goals'].isin(filter_criteria['financial_goals'])]
    
    # Return only the requested columns plus similarity score
    display_cols = ['title', 'financial_goals', 'age_bracket', 'similarity_score']
    return recommendations[display_cols]

def content_generate_ratings(program, user, rating_data, users_with_finance, cosine_sim, k=20, threshold=0.0):
    indices = pd.Series(range(len(users_with_finance)), index=users_with_finance['title']).to_dict()
    if program not in indices.keys():
        return 'Program does not exist'
    idx = indices[program]
    
    programs_rated = rating_data[rating_data['userId'] == user]
    neighbors = []
    
    for _, row in programs_rated.iterrows():
        if row['title'] not in indices:
            continue
        rated_idx = indices[row['title']]
        similarity = cosine_sim[idx, rated_idx]
        neighbors.append((similarity, row['rating']))
    
    # If no programs rated, return overall average
    if not neighbors:
        return np.mean(rating_data[rating_data['title'] == program]['rating']) if not rating_data[rating_data['title'] == program].empty else 0
    
    k_neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:k]
    sim_total, weighted_ratings = 0, 0
    for sim, rating in k_neighbors:
        if sim >= threshold:
            sim_total += sim
            weighted_ratings += sim * rating
    
    if sim_total == 0:
        return np.mean(rating_data[rating_data['title'] == program]['rating']) if not rating_data[rating_data['title'] == program].empty else 0
    
    return weighted_ratings / sim_total

def perform_clustering(users_with_finance, vectorizer, n_clusters=5):
    # Get feature vectors
    program_matrix = vectorizer.transform(users_with_finance['combined_features'])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(program_matrix)
    
    # Add clusters to dataframe
    users_with_finance_clustered = users_with_finance.copy()
    users_with_finance_clustered['cluster'] = clusters
    
    return users_with_finance_clustered, kmeans

def visualize_clusters(clustered_data):
    # Create a PCA projection for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # Get TF-IDF vectors
    vectorizer = TfidfVectorizer()
    program_matrix = vectorizer.fit_transform(clustered_data['combined_features'])
    
    # Apply PCA
    coords = pca.fit_transform(program_matrix.toarray())
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'cluster': clustered_data['cluster'],
        'title': clustered_data['title'],
        'age_bracket': clustered_data['age_bracket'],
        'financial_goals': clustered_data['financial_goals']
    })
    
    # Create interactive plot
    fig = px.scatter(
        plot_df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['title', 'age_bracket', 'financial_goals'],
        color_continuous_scale=px.colors.qualitative.G10
    )
    fig.update_layout(
        title="Program Clusters Visualization",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )
    
    return fig

def analyze_user_preferences(rating_data, user_id):
    user_ratings = rating_data[rating_data['userId'] == user_id]
    
    if user_ratings.empty:
        return None, "No ratings found for this user."
    
    # Calculate statistics
    avg_rating = user_ratings['rating'].mean()
    rating_count = len(user_ratings)
    rating_distribution = user_ratings['rating'].value_counts().sort_index()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(user_ratings['rating'], bins=5, kde=True, ax=ax)
    ax.set_title(f"Rating Distribution for User {user_id}")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    
    return {
        'avg_rating': avg_rating,
        'rating_count': rating_count,
        'top_rated': user_ratings.sort_values('rating', ascending=False).head(5),
        'fig': fig
    }, None

# Streamlit UI
st.title("🎓 Program Recommender System")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Recommendations", "User Analysis", "Program Clusters", "System Analytics"])

# Load data
users_with_finance, rating_data = load_data()
users_with_finance, cosine_sim, vectorizer = preprocess_data(users_with_finance)

if page == "Recommendations":
    st.header("Program Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_program = st.selectbox("Select a Program", users_with_finance['title'].unique())
        N = st.slider("Number of Recommendations", 1, 20, 5)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1, step=0.05)
    
    with col2:
        st.subheader("Filter Options")
        # Get unique values for filters
        age_brackets = users_with_finance['age_bracket'].unique().tolist()
        financial_goals = users_with_finance['financial_goals'].unique().tolist()
        
        selected_age_brackets = st.multiselect("Filter by Age Bracket", age_brackets)
        selected_financial_goals = st.multiselect("Filter by Financial Goals", financial_goals)
        
        filter_criteria = {
            'age_bracket': selected_age_brackets,
            'financial_goals': selected_financial_goals
        }
    
    if st.button("Get Recommendations", key="rec_button"):
        with st.spinner("Finding recommendations..."):
            recommendations = make_recommendations(
                selected_program, 
                N, 
                users_with_finance, 
                cosine_sim, 
                threshold,
                filter_criteria
            )
            
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.success(f"Found {len(recommendations)} recommendations")
                st.dataframe(recommendations)
                
                # Visualization of similarity scores
                if not recommendations.empty:
                    fig = px.bar(
                        recommendations, 
                        x='title', 
                        y='similarity_score',
                        color='similarity_score',
                        labels={'similarity_score': 'Similarity', 'title': 'Program'},
                        title="Similarity Scores of Recommended Programs"
                    )
                    st.plotly_chart(fig)
    
    st.divider()
    
    st.subheader("Predict User Rating")
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
    
    with col2:
        rating_program = st.selectbox("Select Program to Rate", users_with_finance['title'].unique(), key="rating_program")
    
    if st.button("Predict Rating", key="rating_button"):
        with st.spinner("Calculating predicted rating..."):
            predicted_rating = content_generate_ratings(
                rating_program, 
                user_id, 
                rating_data, 
                users_with_finance,
                cosine_sim
            )
            
            if isinstance(predicted_rating, str):
                st.error(predicted_rating)
            else:
                # Create a gauge chart for the rating
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_rating,
                    title = {'text': f"Predicted Rating for {rating_program}"},
                    gauge = {
                        'axis': {'range': [0, 5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgray"},
                            {'range': [2, 3.5], 'color': "gray"},
                            {'range': [3.5, 5], 'color': "darkgray"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig)

elif page == "User Analysis":
    st.header("User Preference Analysis")
    
    user_id = st.number_input("Enter User ID to Analyze", min_value=1, step=1)
    
    if st.button("Analyze User", key="analyze_user"):
        user_data, error = analyze_user_preferences(rating_data, user_id)
        
        if error:
            st.error(error)
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Rating Distribution")
                st.pyplot(user_data['fig'])
            
            with col2:
                st.subheader("User Statistics")
                st.metric("Average Rating", f"{user_data['avg_rating']:.2f}/5.0")
                st.metric("Programs Rated", user_data['rating_count'])
            
            st.subheader("Top Rated Programs")
            st.dataframe(user_data['top_rated'][['title', 'rating']])
            
            # Get recommendations based on top-rated programs
            if not user_data['top_rated'].empty:
                top_program = user_data['top_rated'].iloc[0]['title']
                st.subheader(f"Recommended Programs Based on Your Top Rated: {top_program}")
                
                with st.spinner("Finding similar programs..."):
                    similar_programs = make_recommendations(
                        top_program, 
                        5, 
                        users_with_finance, 
                        cosine_sim
                    )
                    if not isinstance(similar_programs, str):
                        st.dataframe(similar_programs)

elif page == "Program Clusters":
    st.header("Program Clustering Analysis")
    
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    if st.button("Generate Clusters"):
        with st.spinner("Clustering programs..."):
            clustered_data, kmeans = perform_clustering(users_with_finance, vectorizer, n_clusters)
            
            # Visualize clusters
            cluster_fig = visualize_clusters(clustered_data)
            st.plotly_chart(cluster_fig, use_container_width=True)
            
            # Display cluster contents
            st.subheader("Cluster Contents")
            
            tabs = st.tabs([f"Cluster {i}" for i in range(n_clusters)])
            
            for i, tab in enumerate(tabs):
                with tab:
                    cluster_programs = clustered_data[clustered_data['cluster'] == i]
                    st.write(f"**Cluster {i}** - {len(cluster_programs)} programs")
                    
                    # Get most common age brackets and financial goals in this cluster
                    age_counts = cluster_programs['age_bracket'].value_counts()
                    goal_counts = cluster_programs['financial_goals'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top Age Brackets:**")
                        st.dataframe(age_counts.head(3))
                    
                    with col2:
                        st.write("**Top Financial Goals:**")
                        st.dataframe(goal_counts.head(3))
                    
                    st.write("**Programs in this cluster:**")
                    st.dataframe(cluster_programs[['title', 'age_bracket', 'financial_goals']])

elif page == "System Analytics":
    st.header("Recommender System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Program Distribution by Age Bracket")
        age_counts = users_with_finance['age_bracket'].value_counts()
        fig = px.pie(names=age_counts.index, values=age_counts.values, title="Age Bracket Distribution")
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Program Distribution by Financial Goals")
        goal_counts = users_with_finance['financial_goals'].value_counts()
        fig = px.bar(x=goal_counts.index, y=goal_counts.values, title="Financial Goals Distribution")
        st.plotly_chart(fig)
    
    st.subheader("Rating Distribution")
    rating_hist = px.histogram(
        rating_data, 
        x="rating", 
        nbins=10, 
        title="Rating Distribution",
        labels={"rating": "Rating", "count": "Number of Ratings"}
    )
    st.plotly_chart(rating_hist)
    
    # Display correlation matrix of features
    st.subheader("Feature Correlation Analysis")
    
    # Get a sample of programs to analyze
    if len(users_with_finance) > 100:
        sample_data = users_with_finance.sample(100, random_state=42)
    else:
        sample_data = users_with_finance
    
    # Create a correlation matrix for numeric columns
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = sample_data[numeric_cols].corr()
        fig = px.imshow(
            corr, 
            text_auto=True, 
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig)
    else:
        st.info("Not enough numeric columns for correlation analysis")

# Footer
st.markdown("---")
st.markdown("Program Recommender System v2.0")
