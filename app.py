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
from datetime import datetime, timedelta
import altair as alt
from streamlit_option_menu import option_menu
import random
from wordcloud import WordCloud
from collections import Counter
import re

# App configuration
st.set_page_config(layout="wide", page_title="Program Recommender System", page_icon="🎓")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    users_with_finance = pd.read_csv("users_with_finance.csv")
    rating_data = pd.read_csv("rating_data.csv")
    
    # Simulate additional data for new features
    # Add engagement data
    if 'engagement_score' not in users_with_finance.columns:
        users_with_finance['engagement_score'] = np.random.uniform(1, 10, len(users_with_finance))
    if 'completion_rate' not in users_with_finance.columns:
        users_with_finance['completion_rate'] = np.random.uniform(0.3, 1.0, len(users_with_finance))
    if 'total_enrollments' not in users_with_finance.columns:
        users_with_finance['total_enrollments'] = np.random.randint(50, 1000, len(users_with_finance))
    
    # Add time data to ratings if not present
    if 'timestamp' not in rating_data.columns:
        # Generate random timestamps within the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        rating_data['timestamp'] = [start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))) 
            for _ in range(len(rating_data))]
    elif not isinstance(rating_data['timestamp'].iloc[0], datetime):
        rating_data['timestamp'] = pd.to_datetime(rating_data['timestamp'])
    
    # Add trending scores based on recent ratings
    if 'trending_score' not in users_with_finance.columns:
        users_with_finance['trending_score'] = np.random.uniform(0, 100, len(users_with_finance))
    
    # Add topic keywords if not present
    if 'keywords' not in users_with_finance.columns:
        topics = ['investment', 'saving', 'budgeting', 'retirement', 'stocks', 
                 'bonds', 'real estate', 'taxes', 'insurance', 'debt management',
                 'financial planning', 'wealth building', 'passive income']
        users_with_finance['keywords'] = [', '.join(random.sample(topics, random.randint(2, 5))) 
                                         for _ in range(len(users_with_finance))]
    
    # Add difficulty level if not present
    if 'difficulty' not in users_with_finance.columns:
        difficulties = ['Beginner', 'Intermediate', 'Advanced']
        users_with_finance['difficulty'] = [random.choice(difficulties) for _ in range(len(users_with_finance))]
    
    return users_with_finance, rating_data

def preprocess_data(users_with_finance):
    # If combined_features doesn't exist, create it
    if 'combined_features' not in users_with_finance.columns:
        # Combine text features for vectorization
        text_cols = ['title', 'financial_goals', 'age_bracket']
        if 'keywords' in users_with_finance.columns:
            text_cols.append('keywords')
        if 'difficulty' in users_with_finance.columns:
            text_cols.append('difficulty')
        
        users_with_finance['combined_features'] = users_with_finance[text_cols].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    program_matrix = vectorizer.fit_transform(users_with_finance['combined_features'])
    cosine_sim = cosine_similarity(program_matrix, program_matrix)
    return users_with_finance, cosine_sim, vectorizer

def hybrid_recommendations(program, user_id, N, users_with_finance, rating_data, cosine_sim, 
                           content_weight=0.7, collaborative_weight=0.3, trending_weight=0.1,
                           engagement_weight=0.1, recency_weight=0.1, threshold=0.0,
                           filter_criteria=None):
    """
    Generates hybrid recommendations using content, collaborative filtering,
    trending scores, engagement, and recency.
    """
    # Base content recommendations
    content_recs = make_recommendations(program, N*2, users_with_finance, cosine_sim, threshold, filter_criteria, 
                                       return_indices=True)
    
    if isinstance(content_recs, str):
        return content_recs
    
    # Get all recommended program indices
    content_indices = content_recs['index'].tolist()
    
    # Get content similarity scores (normalized)
    content_scores = {idx: score for idx, score in zip(content_recs['index'], content_recs['similarity_score'])}
    
    # Get collaborative filtering scores for the user if they have ratings
    collab_scores = {}
    if user_id is not None:
        user_ratings = rating_data[rating_data['userId'] == user_id]
        if not user_ratings.empty:
            for idx in content_indices:
                prog_title = users_with_finance.iloc[idx]['title']
                pred_rating = content_generate_ratings(prog_title, user_id, rating_data, users_with_finance, cosine_sim)
                if not isinstance(pred_rating, str):
                    # Normalize rating to 0-1 scale (from 1-5 scale)
                    collab_scores[idx] = (pred_rating - 1) / 4
    
    # Get trending scores (normalized)
    trending_scores = {}
    for idx in content_indices:
        if 'trending_score' in users_with_finance.columns:
            trending_scores[idx] = users_with_finance.iloc[idx]['trending_score'] / 100
    
    # Get engagement scores (normalized)
    engagement_scores = {}
    for idx in content_indices:
        if 'engagement_score' in users_with_finance.columns and 'completion_rate' in users_with_finance.columns:
            engagement_score = users_with_finance.iloc[idx]['engagement_score'] / 10  # Normalize 1-10 to 0-1
            completion_rate = users_with_finance.iloc[idx]['completion_rate']  # Already 0-1
            engagement_scores[idx] = (engagement_score + completion_rate) / 2
    
    # Get recency scores (optional - reward newer items or recently rated items)
    recency_scores = {}
    if 'timestamp' in rating_data.columns:
        recent_ratings = rating_data.sort_values('timestamp', ascending=False).head(100)
        recent_titles = recent_ratings['title'].value_counts().to_dict()
        max_count = max(recent_titles.values()) if recent_titles else 1
        
        for idx in content_indices:
            prog_title = users_with_finance.iloc[idx]['title']
            if prog_title in recent_titles:
                # Normalize by max count
                recency_scores[idx] = recent_titles[prog_title] / max_count
            else:
                recency_scores[idx] = 0
    
    # Combine scores with weights
    final_scores = {}
    for idx in content_indices:
        score = content_weight * content_scores.get(idx, 0)
        if idx in collab_scores:
            score += collaborative_weight * collab_scores.get(idx, 0)
        if idx in trending_scores:
            score += trending_weight * trending_scores.get(idx, 0)
        if idx in engagement_scores:
            score += engagement_weight * engagement_scores.get(idx, 0)
        if idx in recency_scores:
            score += recency_weight * recency_scores.get(idx, 0)
        final_scores[idx] = score
    
    # Get top N programs
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:N]
    top_indices = [idx for idx, _ in sorted_scores]
    top_scores = [score for _, score in sorted_scores]
    
    # Create final recommendations dataframe
    recommendations = users_with_finance.iloc[top_indices].reset_index(drop=True)
    recommendations['hybrid_score'] = top_scores
    
    # Add component scores for transparency
    recommendations['content_score'] = [content_scores.get(idx, 0) for idx in top_indices]
    recommendations['collaborative_score'] = [collab_scores.get(idx, 0) if idx in collab_scores else np.nan for idx in top_indices]
    recommendations['trending_score'] = [trending_scores.get(idx, 0) if idx in trending_scores else np.nan for idx in top_indices]
    recommendations['engagement_score'] = [engagement_scores.get(idx, 0) if idx in engagement_scores else np.nan for idx in top_indices]
    recommendations['recency_score'] = [recency_scores.get(idx, 0) if idx in recency_scores else np.nan for idx in top_indices]
    
    # Determine which columns to display
    display_cols = ['title', 'financial_goals', 'age_bracket', 'hybrid_score']
    if 'difficulty' in recommendations.columns:
        display_cols.insert(3, 'difficulty')
    if 'keywords' in recommendations.columns:
        display_cols.insert(len(display_cols)-1, 'keywords')
    
    return recommendations[display_cols]

def make_recommendations(program, N, users_with_finance, cosine_sim, threshold=0.0, filter_criteria=None, 
                         return_indices=False):
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
    
    # Add original indices if requested (for hybrid recommendations)
    if return_indices:
        recommendations['index'] = program_idx
    
    # Apply filters if provided
    if filter_criteria and 'age_bracket' in filter_criteria and filter_criteria['age_bracket']:
        recommendations = recommendations[recommendations['age_bracket'].isin(filter_criteria['age_bracket'])]
    if filter_criteria and 'financial_goals' in filter_criteria and filter_criteria['financial_goals']:
        recommendations = recommendations[recommendations['financial_goals'].isin(filter_criteria['financial_goals'])]
    if filter_criteria and 'difficulty' in filter_criteria and filter_criteria['difficulty'] and 'difficulty' in recommendations.columns:
        recommendations = recommendations[recommendations['difficulty'].isin(filter_criteria['difficulty'])]
    
    # Return only the requested columns plus similarity score
    display_cols = ['title', 'financial_goals', 'age_bracket', 'similarity_score']
    if 'difficulty' in recommendations.columns:
        display_cols.insert(3, 'difficulty')
    if 'keywords' in recommendations.columns:
        display_cols.insert(len(display_cols)-1, 'keywords')
    if return_indices:
        display_cols.append('index')
    
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
    vectorizer = TfidfVectorizer(stop_words='english')
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
    
    # Add temporal analysis if timestamp exists
    if 'timestamp' in user_ratings.columns:
        user_ratings = user_ratings.sort_values('timestamp')
        user_ratings['month'] = user_ratings['timestamp'].dt.to_period('M')
        monthly_ratings = user_ratings.groupby('month')['rating'].mean()
        
        # Convert Period to datetime for plotting
        monthly_data = pd.DataFrame({
            'month': [pd.to_datetime(str(p)) for p in monthly_ratings.index],
            'avg_rating': monthly_ratings.values
        })
        
        time_fig = px.line(
            monthly_data, 
            x='month', 
            y='avg_rating',
            title=f"Average Monthly Ratings for User {user_id}",
            labels={'month': 'Month', 'avg_rating': 'Average Rating'}
        )
    else:
        time_fig = None
    
    return {
        'avg_rating': avg_rating,
        'rating_count': rating_count,
        'top_rated': user_ratings.sort_values('rating', ascending=False).head(5),
        'fig': fig,
        'time_fig': time_fig
    }, None

def generate_user_persona(user_id, users_with_finance, rating_data):
    """Generate a user persona based on their ratings and interactions"""
    user_ratings = rating_data[rating_data['userId'] == user_id]
    
    if user_ratings.empty:
        return None, "No ratings found for this user."
    
    # Get top rated programs
    top_rated = user_ratings.sort_values('rating', ascending=False).head(10)
    
    # Extract features from top rated programs
    age_brackets = []
    financial_goals = []
    keywords = []
    difficulties = []
    
    for _, row in top_rated.iterrows():
        program = users_with_finance[users_with_finance['title'] == row['title']]
        if not program.empty:
            age_brackets.append(program['age_bracket'].iloc[0])
            financial_goals.append(program['financial_goals'].iloc[0])
            if 'keywords' in program.columns:
                keywords.extend([k.strip() for k in program['keywords'].iloc[0].split(',')])
            if 'difficulty' in program.columns:
                difficulties.append(program['difficulty'].iloc[0])
    
    # Get most common values
    top_age_bracket = Counter(age_brackets).most_common(1)[0][0] if age_brackets else "Unknown"
    top_goals = [g[0] for g in Counter(financial_goals).most_common(3)] if financial_goals else ["Unknown"]
    top_difficulty = Counter(difficulties).most_common(1)[0][0] if difficulties else "Unknown"
    
    # Create keyword cloud if keywords exist
    if keywords:
        keyword_text = ' '.join(keywords)
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             min_font_size=10).generate(keyword_text)
        # Convert to image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
    else:
        fig = None
    
    # Calculate engagement metrics
    activity_level = "High" if len(user_ratings) > 20 else "Medium" if len(user_ratings) > 10 else "Low"
    
    # Get rating trend if timestamps available
    trend = None
    if 'timestamp' in user_ratings.columns and len(user_ratings) >= 5:
        user_ratings = user_ratings.sort_values('timestamp')
        from scipy.stats import linregress
        x = range(len(user_ratings))
        y = user_ratings['rating'].values
        slope, _, _, _, _ = linregress(x, y)
        trend = "Increasing" if slope > 0.1 else "Decreasing" if slope < -0.1 else "Stable"
    
    return {
        'user_id': user_id,
        'top_age_bracket': top_age_bracket,
        'top_goals': top_goals,
        'top_difficulty': top_difficulty,
        'activity_level': activity_level,
        'rating_trend': trend,
        'wordcloud_fig': fig,
        'rating_count': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'rating_variance': user_ratings['rating'].var()
    }, None

def analyze_trending_programs(rating_data, users_with_finance, days=30):
    """Analyze trending programs based on recent ratings"""
    if 'timestamp' not in rating_data.columns:
        return None, "Timestamp data not available for trend analysis."
    
    # Filter recent ratings
    if isinstance(rating_data['timestamp'].iloc[0], str):
        rating_data['timestamp'] = pd.to_datetime(rating_data['timestamp'])
    
    recent_cutoff = datetime.now() - timedelta(days=days)
    recent_ratings = rating_data[rating_data['timestamp'] > recent_cutoff]
    
    if recent_ratings.empty:
        return None, f"No ratings found in the last {days} days."
    
    # Count ratings per program
    program_counts = recent_ratings['title'].value_counts().reset_index()
    program_counts.columns = ['title', 'recent_ratings']
    
    # Get average rating per program
    program_avg = recent_ratings.groupby('title')['rating'].mean().reset_index()
    program_avg.columns = ['title', 'recent_avg_rating']
    
    # Combine counts and averages
    trending = pd.merge(program_counts, program_avg, on='title')
    
    # Calculate a trending score (combination of count and average rating)
    max_count = trending['recent_ratings'].max()
    trending['trending_score'] = (trending['recent_ratings'] / max_count * 0.7) + (trending['recent_avg_rating'] / 5 * 0.3)
    trending = trending.sort_values('trending_score', ascending=False)
    
    # Merge with program data
    trending_with_data = pd.merge(trending, users_with_finance, on='title', how='left')
    
    return trending_with_data.head(20), None

def get_program_details(program_title, users_with_finance, rating_data):
    """Get detailed information about a specific program"""
    if program_title not in users_with_finance['title'].values:
        return None, "Program not found."
    
    program_data = users_with_finance[users_with_finance['title'] == program_title].iloc[0]
    
    # Get ratings for this program
    program_ratings = rating_data[rating_data['title'] == program_title]
    
    # Calculate rating statistics
    if not program_ratings.empty:
        avg_rating = program_ratings['rating'].mean()
        rating_count = len(program_ratings)
        rating_dist = program_ratings['rating'].value_counts().sort_index()
        
        # Rating distribution chart
        fig = px.bar(
            x=rating_dist.index, 
            y=rating_dist.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title=f"Rating Distribution for {program_title}"
        )
        
        # Rating over time if timestamps available
        if 'timestamp' in program_ratings.columns and len(program_ratings) >= 5:
            program_ratings = program_ratings.sort_values('timestamp')
            program_ratings['month'] = program_ratings['timestamp'].dt.to_period('M')
            monthly_ratings = program_ratings.groupby('month')['rating'].mean().reset_index()
            monthly_ratings['month'] = monthly_ratings['month'].astype(str)
            
            time_fig = px.line(
                monthly_ratings,
                x='month',
                y='rating',
                title=f"Average Monthly Ratings for {program_title}"
            )
        else:
            time_fig = None
    else:
        avg_rating = None
        rating_count = 0
        fig = None
        time_fig = None
    
    # Extract keywords if available
    keywords = []
    if 'keywords' in program_data and not pd.isna(program_data['keywords']):
        keywords = [k.strip() for k in program_data['keywords'].split(',')]
    
    return {
        'program_data': program_data,
        'avg_rating': avg_rating,
        'rating_count': rating_count,
        'rating_fig': fig,
        'time_fig': time_fig,
        'keywords': keywords
    }, None

def recommend_for_new_users(users_with_finance, rating_data, n=5):
    """Recommend programs for new users based on popularity and quality"""
    # Get average ratings per program
    program_ratings = rating_data.groupby('title')['rating'].agg(['mean', 'count']).reset_index()
    program_ratings.columns = ['title', 'avg_rating', 'num_ratings']
    
    # Calculate a score combining popularity and quality
    # Using a weighted version of the IMDB weighted rating formula
    C = program_ratings['avg_rating'].mean()  # Mean rating across all programs
    m = 5  # Minimum ratings required (can be adjusted)
    program_ratings['score'] = (program_ratings['num_ratings'] / (program_ratings['num_ratings'] + m) * 
                              program_ratings['avg_rating'] + m / (program_ratings['num_ratings'] + m) * C)
    
    # Sort by score
    program_ratings = program_ratings.sort_values('score', ascending=False)
    
    # Get top programs
    top_programs = program_ratings.head(n)
    
    # Merge with program details
    recommendations = pd.merge(top_programs, users_with_finance, on='title', how='left')
    
    # Select display columns
    display_cols = ['title', 'financial_goals', 'age_bracket', 'avg_rating', 'num_ratings']
    if 'difficulty' in recommendations.columns:
        display_cols.insert(3, 'difficulty')
    
    return recommendations[display_cols]

def generate_content_based_profile(user_id, users_with_finance, rating_data, vectorizer):
    """Generate a content-based profile for a user based on their ratings"""
    user_ratings = rating_data[rating_data['userId'] == user_id]
    
    if user_ratings.empty:
        return None, "No ratings found for this user."
    
    # Get programs rated by this user
    rated_programs = []
    for _, row in user_ratings.iterrows():
        program_data = users_with_finance[users_with_finance['title'] == row['title']]
        if not program_data.empty:
            rated_programs.append({
                'title': row['title'],
                'rating': row['rating'],
                'index': program_data.index[0]
            })
    
    if not rated_programs:
        return None, "None of the rated programs found in the database."
    
    # Create a user profile based on the TF-IDF vectors of rated programs
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = vectorizer.transform(users_with_finance['combined_features'])
    
    # Initialize user profile
    user_profile = np.zeros(len(feature_names))
    
    # Weight each program by user rating (normalized to 0-1)
    rating_sum = 0
    for program in rated_programs:
        rating_weight = (program['rating'] - 1) / 4  # Normalize from 1-5 to 0-1
        program_vector = tfidf_matrix[program['index']].toarray().flatten()
        user_profile += program_vector * rating_weight
        rating_sum += rating_weight
    
    # Normalize user profile
    if rating_sum > 0:
        user_profile = user_profile / rating_sum
    
    # Get top features in user profile
    top_indices = np.argsort(user_profile)[-20:][::-1]
    top_features = [(feature_names[i], user_profile[i]) for i in top_indices if user_profile[i] > 0]
    
    # Create wordcloud from top features
    if top_features:
        wordcloud_dict = {feature: weight for feature, weight in top_features}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_dict)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
    else:
        fig = None
    
    return {
        'user_id': user_id,
        'top_features': top_features,
        'profile_vector': user_profile,
        'wordcloud_fig': fig
    }, None

def export_recommendations_to_csv(recommendations, format='csv'):
    """Export recommendations to a CSV file for download"""
    if format == 'csv':
        csv = recommendations.to_csv(index=False)
        return csv
    elif format == 'excel':
        # For Excel, we return a CSV but with a different file extension
        csv = recommendations.to_csv(index=False)
        return csv
    else:
        return None

# Main application
def main():
    # Load data
    users_with_finance, rating_data = load_data()
    users_with_finance, cosine_sim, vectorizer = preprocess_data(users_with_finance)
    
    # Custom sidebar navigation with icons
    with st.sidebar:
        st.image("https://www.example.com/logo.png", width=100)  # Replace with actual logo URL
        st.title("Navigation")
        
        # Use streamlit-option-menu for better UI
        try:
            selected = option_menu(
                menu_title=None,
                options=[
                    "Home", "Recommendations", "User Analysis", 
                    "Program Explorer", "Trending Programs", "Analytics"
                ],
                icons=[
                    "house", "search", "person", 
                    "collection", "graph-up", "bar-chart"
                ],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5px", "background-color": "#f0f2f6"},
                    "icon": {"color": "#1E88E5", "font-size": "16px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#1E88E5", "color": "white"},
                }
            )
        except:
            selected = st.sidebar.radio(
                "Navigate to:",
                ["Home", "Recommendations", "User Analysis", "Program Explorer", "Trending Programs", "Analytics"]
            )
        
        st.sidebar.divider()
        st.sidebar.markdown("### Filters")
        
        # Global filters
        age_brackets = sorted(users_with_finance['age_bracket'].unique())
        financial_goals = sorted(users_with_finance['financial_goals'].unique())
        
        selected_age_brackets = st.sidebar.multiselect("Age Bracket", age_brackets)
        selected_goals = st.sidebar.multiselect("Financial Goals", financial_goals)
        
        if 'difficulty' in users_with_finance.columns:
            difficulties = sorted(users_with_finance['difficulty'].unique())
            selected_difficulties = st.sidebar.multiselect("Difficulty Level", difficulties)
        else:
            selected_difficulties = []
        
        filter_criteria = {
            'age_bracket': selected_age_brackets,
            'financial_goals': selected_goals,
            'difficulty': selected_difficulties
        }
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("© 2023 Program Recommender")
    
    # Home page
    if selected == "Home":
        st.markdown("<h1 class='main-header'>Financial Education Program Recommender</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Welcome to the Financial Education Program Recommender! This intelligent system helps you discover
            the most relevant financial education programs based on your needs and preferences.
            
            ### How to use this app:
            
            - **Recommendations**: Get personalized program recommendations
            - **User Analysis**: View your profile and preferences
            - **Program Explorer**: Browse and search all available programs
            - **Trending Programs**: See what's popular right now
            - **Analytics**: Explore insights and patterns
            
            Use the sidebar to navigate between different sections and apply filters.
            """)
        
        with col2:
            st.image("https://www.example.com/finance_image.jpg", caption="")  # Replace with actual image URL
        
        # Quick stats
        st.markdown("<h2 class='sub-header'>Program Statistics</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Total Programs", len(users_with_finance))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Total Ratings", len(rating_data))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            avg_rating = round(rating_data['rating'].mean(), 2)
            st.metric("Average Rating", f"{avg_rating}/5")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Active Users", rating_data['userId'].nunique())
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Top programs
        st.markdown("<h2 class='sub-header'>Top Rated Programs</h2>", unsafe_allow_html=True)
        top_programs = rating_data.groupby('title')['rating'].agg(['mean', 'count'])
        top_programs = top_programs[top_programs['count'] >= 5].sort_values('mean', ascending=False).head(5)
        top_programs = top_programs.reset_index()
        
        fig = px.bar(
            top_programs, 
            x='title', 
            y='mean', 
            text='mean',
            hover_data=['count'],
            labels={'title': 'Program', 'mean': 'Average Rating', 'count': 'Number of Ratings'},
            title="Top 5 Highest Rated Programs (min. 5 ratings)",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Featured program
        st.markdown("<h2 class='sub-header'>Featured Program</h2>", unsafe_allow_html=True)
        featured_idx = np.random.randint(0, len(users_with_finance))
        featured_program = users_with_finance.iloc[featured_idx]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://img.icons8.com/?size=100&id=WFSwkOenGPSn&format=png&color=000000", width=200)  # Replace with actual image URL
        
        with col2:
            st.markdown(f"### {featured_program['title']}")
            st.markdown(f"**Target Age Group:** {featured_program['age_bracket']}")
            st.markdown(f"**Financial Goal:** {featured_program['financial_goals']}")
            if 'difficulty' in featured_program:
                st.markdown(f"**Difficulty:** {featured_program['difficulty']}")
            if 'keywords' in featured_program:
                st.markdown(f"**Topics:** {featured_program['keywords']}")
    
    # Recommendations page
    elif selected == "Recommendations":
        st.markdown("<h1 class='main-header'>Program Recommendations</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        Get personalized recommendations based on your preferences. Choose a program you like as a starting point,
        or select your user ID to get recommendations tailored to your rating history.
        """)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            program_list = sorted(users_with_finance['title'].unique())
            selected_program = st.selectbox(
                "Select a program you like:",
                program_list
            )
        
        with col2:
            user_ids = sorted(rating_data['userId'].unique())
            selected_user = st.selectbox(
                "Or select your user ID (optional):",
                ["None"] + [str(id) for id in user_ids]
            )
            selected_user = None if selected_user == "None" else int(selected_user)
        
        with col3:
            num_recs = st.slider("Number of recommendations:", 3, 20, 5)
        
        # Advanced options (collapsible)
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                content_weight = st.slider("Content Similarity Weight:", 0.0, 1.0, 0.7, 0.1)
                collab_weight = st.slider("User Preference Weight:", 0.0, 1.0, 0.3, 0.1)
                
            with col2:
                trending_weight = st.slider("Trending Weight:", 0.0, 1.0, 0.1, 0.1)
                engagement_weight = st.slider("Engagement Weight:", 0.0, 1.0, 0.1, 0.1)
                recency_weight = st.slider("Recency Weight:", 0.0, 1.0, 0.1, 0.1)
            
            similarity_threshold = st.slider("Minimum Similarity Threshold:", 0.0, 1.0, 0.0, 0.1)
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                if selected_program:
                    recommendations = hybrid_recommendations(
                        selected_program, selected_user, num_recs, 
                        users_with_finance, rating_data, cosine_sim,
                        content_weight, collab_weight, trending_weight,
                        engagement_weight, recency_weight, similarity_threshold,
                        filter_criteria
                    )
                    
                    if isinstance(recommendations, str):
                        st.error(recommendations)
                    else:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        st.dataframe(recommendations.style.background_gradient(subset=['hybrid_score'], cmap='Blues'))
                        
                        # Export options
                        if len(recommendations) > 0:
                            csv = export_recommendations_to_csv(recommendations)
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=csv,
                                file_name="program_recommendations.csv",
                                mime="text/csv",
                            )
                        
                        # Explanation of recommendation factors
                        st.info("""
                        **How these recommendations were generated:**
                        - Content Similarity: Programs with similar topics and target audience
                        - User Preference: Based on your rating history (if user ID selected)
                        - Trending: Currently popular programs
                        - Engagement: Programs with high completion rates
                        - Recency: Recently popular programs
                        """)
                else:
                    st.error("Please select a program to get recommendations.")
        
        # If no recommendations yet, show some for new users
        if 'recommendations' not in locals():
            st.markdown("<h3 class='sub-header'>Popular Programs for New Users</h3>", unsafe_allow_html=True)
            popular_recs = recommend_for_new_users(users_with_finance, rating_data, n=5)
            st.dataframe(popular_recs)
    
    # User Analysis page
    elif selected == "User Analysis":
        st.markdown("<h1 class='main-header'>User Analysis</h1>", unsafe_allow_html=True)
        
        user_ids = sorted(rating_data['userId'].unique())
        selected_user = st.selectbox(
            "Select a user ID to analyze:",
            [str(id) for id in user_ids]
        )
        selected_user = int(selected_user)
        
        tab1, tab2, tab3 = st.tabs(["User Profile", "Preferences", "Content Profile"])
        
        with tab1:
            with st.spinner("Generating user profile..."):
                profile, error = generate_user_persona(selected_user, users_with_finance, rating_data)
                
                if error:
                    st.error(error)
                elif profile:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"### User {profile['user_id']}")
                        st.markdown(f"**Preferred Age Bracket:** {profile['top_age_bracket']}")
                        st.markdown(f"**Top Financial Goals:**")
                        for goal in profile['top_goals']:
                            st.markdown(f"- {goal}")
                        
                        if profile['top_difficulty'] != "Unknown":
                            st.markdown(f"**Preferred Difficulty:** {profile['top_difficulty']}")
                        
                        st.markdown(f"**Activity Level:** {profile['activity_level']}")
                        st.markdown(f"**Total Ratings:** {profile['rating_count']}")
                        st.markdown(f"**Average Rating:** {profile['avg_rating']:.2f}/5")
                        
                        if profile['rating_trend']:
                            st.markdown(f"**Rating Trend:** {profile['rating_trend']}")
                    
                    with col2:
                        if profile['wordcloud_fig']:
                            st.pyplot(profile['wordcloud_fig'])
                            st.caption("Word cloud of topics from user's top-rated programs")
        
        with tab2:
            with st.spinner("Analyzing user preferences..."):
                preferences, error = analyze_user_preferences(rating_data, selected_user)
                
                if error:
                    st.error(error)
                elif preferences:
                    st.markdown(f"### Rating Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Rating", f"{preferences['avg_rating']:.2f}/5")
                    with col2:
                        st.metric("Total Ratings", preferences['rating_count'])
                    with col3:
                        st.metric("Rating Variance", f"{preferences['top_rated']['rating'].var():.2f}")
                    
                    st.markdown("### Top Rated Programs")
                    st.dataframe(preferences['top_rated'][['title', 'rating']])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.pyplot(preferences['fig'])
                    
                    with col2:
                        if preferences['time_fig']:
                            st.plotly_chart(preferences['time_fig'], use_container_width=True)
        
        with tab3:
            with st.spinner("Generating content profile..."):
                content_profile, error = generate_content_based_profile(selected_user, users_with_finance, rating_data, vectorizer)
                
                if error:
                    st.error(error)
                elif content_profile:
                    st.markdown("### Content Interests")
                    
                    if content_profile['wordcloud_fig']:
                        st.pyplot(content_profile['wordcloud_fig'])
                        st.caption("Word cloud representing user's content preferences")
                    
                    st.markdown("### Top Interest Terms")
                    terms_df = pd.DataFrame(content_profile['top_features'], columns=['Term', 'Weight'])
                    terms_df = terms_df.sort_values('Weight', ascending=False).head(10)
                    
                    fig = px.bar(
                        terms_df, 
                        x='Weight', 
                        y='Term',
                        orientation='h',
                        labels={'Weight': 'Interest Level', 'Term': 'Topic/Term'},
                        title="Top Interest Terms"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations based on content profile
                    st.markdown("### Recommended Programs Based on Content Profile")
                    # Placeholder for content-based recommendations
                    # This would use the user profile vector to recommend programs
    
    # Program Explorer page
    elif selected == "Program Explorer":
        st.markdown("<h1 class='main-header'>Program Explorer</h1>", unsafe_allow_html=True)
        
        # Search and filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search programs by keyword:", "")
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Relevance", "Rating", "Popularity"]
            )
        
        # Apply filters from sidebar
        filtered_programs = users_with_finance.copy()
        
        if filter_criteria['age_bracket']:
            filtered_programs = filtered_programs[filtered_programs['age_bracket'].isin(filter_criteria['age_bracket'])]
        
        if filter_criteria['financial_goals']:
            filtered_programs = filtered_programs[filtered_programs['financial_goals'].isin(filter_criteria['financial_goals'])]
        
        if filter_criteria['difficulty'] and 'difficulty' in filtered_programs.columns:
            filtered_programs = filtered_programs[filtered_programs['difficulty'].isin(filter_criteria['difficulty'])]
        
        # Apply search term
        if search_term:
            # Search in title, financial_goals, and keywords if available
            search_cols = ['title', 'financial_goals']
            if 'keywords' in filtered_programs.columns:
                search_cols.append('keywords')
            
            search_mask = filtered_programs.apply(
                lambda row: any(search_term.lower() in str(row[col]).lower() for col in search_cols),
                axis=1
            )
            filtered_programs = filtered_programs[search_mask]
        
        # Sort programs
        if sort_by == "Rating":
            # Merge with rating data
            prog_ratings = rating_data.groupby('title')['rating'].mean().reset_index()
            filtered_programs = pd.merge(filtered_programs, prog_ratings, on='title', how='left')
            filtered_programs = filtered_programs.sort_values('rating', ascending=False)
        elif sort_by == "Popularity":
            # Sort by number of ratings
            prog_counts = rating_data['title'].value_counts().reset_index()
            prog_counts.columns = ['title', 'count']
            filtered_programs = pd.merge(filtered_programs, prog_counts, on='title', how='left')
            filtered_programs = filtered_programs.sort_values('count', ascending=False)
        
        # Display programs
        st.markdown(f"### Found {len(filtered_programs)} Programs")
        
        # Display programs in a grid
        cols_per_row = 3
        rows = (len(filtered_programs) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                program_idx = row * cols_per_row + col_idx
                if program_idx < len(filtered_programs):
                    program = filtered_programs.iloc[program_idx]
                    with cols[col_idx]:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"#### {program['title']}")
                        st.markdown(f"**Target:** {program['age_bracket']}")
                        st.markdown(f"**Goal:** {program['financial_goals']}")
                        if 'difficulty' in program:
                            st.markdown(f"**Difficulty:** {program['difficulty']}")
                        
                        # Get rating if available
                        prog_ratings = rating_data[rating_data['title'] == program['title']]
                        if not prog_ratings.empty:
                            avg_rating = prog_ratings['rating'].mean()
                            num_ratings = len(prog_ratings)
                            st.markdown(f"**Rating:** {avg_rating:.1f}/5 ({num_ratings} ratings)")
                        
                        # Button to view details
                        if st.button(f"View Details", key=f"details_{program_idx}"):
                            st.session_state.selected_program = program['title']
                            st.session_state.view_details = True
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Program details view
        if 'view_details' in st.session_state and st.session_state.view_details:
            program_title = st.session_state.selected_program
            details, error = get_program_details(program_title, users_with_finance, rating_data)
            
            if error:
                st.error(error)
            elif details:
                st.markdown("<h2 class='sub-header'>Program Details</h2>", unsafe_allow_html=True)
                
                program_data = details['program_data']
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image("https://www.example.com/placeholder.jpg", width=300)  # Replace with actual image URL
                
                with col2:
                    st.markdown(f"### {program_data['title']}")
                    st.markdown(f"**Target Age Group:** {program_data['age_bracket']}")
                    st.markdown(f"**Financial Goal:** {program_data['financial_goals']}")
                    
                    if 'difficulty' in program_data:
                        st.markdown(f"**Difficulty:** {program_data['difficulty']}")
                    
                    if details['keywords']:
                        st.markdown("**Topics:**")
                        st.markdown(", ".join(details['keywords']))
                    
                    if details['avg_rating']:
                        st.markdown(f"**Average Rating:** {details['avg_rating']:.2f}/5 ({details['rating_count']} ratings)")
                
                # Rating distribution
                if details['rating_fig']:
                    st.plotly_chart(details['rating_fig'], use_container_width=True)
                
                # Ratings over time
                if details['time_fig']:
                    st.plotly_chart(details['time_fig'], use_container_width=True)
                
                # Similar programs
                st.markdown("### Similar Programs")
                similar_programs = make_recommendations(program_title, 5, users_with_finance, cosine_sim)
                if not isinstance(similar_programs, str):
                    st.dataframe(similar_programs)
                
                if st.button("Back to Program List"):
                    st.session_state.view_details = False
                    st.experimental_rerun()
    
    # Trending Programs page
    elif selected == "Trending Programs":
        st.markdown("<h1 class='main-header'>Trending Programs</h1>", unsafe_allow_html=True)
        
        time_period = st.selectbox(
            "Trending over what time period?",
            ["Last 7 days", "Last 30 days", "Last 90 days"]
        )
        
        days = 7 if time_period == "Last 7 days" else 30 if time_period == "Last 30 days" else 90
        
        with st.spinner(f"Analyzing trends over the {time_period}..."):
            trending_data, error = analyze_trending_programs(rating_data, users_with_finance, days)
            
            if error:
                st.error(error)
            elif trending_data is not None:
                # Top trending chart
                st.markdown(f"### Top Trending Programs ({time_period})")
                
                fig = px.bar(
                    trending_data.head(10), 
                    x='trending_score_x', 
                    y='title',
                    orientation='h',
                    labels={'trending_score_x': 'Trending Score', 'title': 'Program'},
                    hover_data=['recent_ratings', 'recent_avg_rating'],
                    color='trending_score_x',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Trending programs by category
                st.markdown("### Trending by Financial Goal")
                
                goal_trends = trending_data.groupby('financial_goals')['trending_score_x'].mean().reset_index()
                goal_trends = goal_trends.sort_values('trending_score_x', ascending=False)
                
                fig = px.pie(
                    goal_trends, 
                    values='trending_score_x', 
                    names='financial_goals',
                    title="Trending Score Distribution by Financial Goal"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display trending programs
                st.markdown("### All Trending Programs")
                st.dataframe(
                    trending_data[['title', 'recent_ratings', 'recent_avg_rating', 'trending_score_x', 
                                  'financial_goals', 'age_bracket']]
                )
    
    # Analytics page
    elif selected == "Analytics":
        st.markdown("<h1 class='main-header'>Program Analytics</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Program Clusters", "Rating Analysis", "User Demographics"])
        
        with tab1:
            st.markdown("### Program Clustering Analysis")
            st.markdown("""
            This analysis groups similar programs together based on their content and target audience.
            Explore the clusters to understand the patterns in our program offerings.
            """)
            
            num_clusters = st.slider("Number of Clusters:", 3, 10, 5)
            
            with st.spinner("Performing clustering analysis..."):
                clustered_data, kmeans = perform_clustering(users_with_finance, vectorizer, num_clusters)
                cluster_viz = visualize_clusters(clustered_data)
                
                st.plotly_chart(cluster_viz, use_container_width=True)
                
                # Cluster analysis
                st.markdown("### Cluster Characteristics")
                
                for i in range(num_clusters):
                    cluster_programs = clustered_data[clustered_data['cluster'] == i]
                    
                    with st.expander(f"Cluster {i+1} ({len(cluster_programs)} programs)"):
                        # Most common age brackets
                        age_counts = cluster_programs['age_bracket'].value_counts()
                        st.markdown(f"**Common Age Brackets:** {', '.join(age_counts.index[:3])}")
                        
                        # Most common financial goals
                        goal_counts = cluster_programs['financial_goals'].value_counts()
                        st.markdown(f"**Common Financial Goals:** {', '.join(goal_counts.index[:3])}")
                        
                        # Extract common keywords if available
                        if 'keywords' in cluster_programs.columns:
                            all_keywords = []
                            for keywords in cluster_programs['keywords']:
                                if isinstance(keywords, str):
                                    all_keywords.extend([k.strip() for k in keywords.split(',')])
                            
                            keyword_counts = Counter(all_keywords)
                            common_keywords = [k for k, _ in keyword_counts.most_common(5)]
                            st.markdown(f"**Common Topics:** {', '.join(common_keywords)}")
                        
                        # Sample programs
                        st.markdown("**Sample Programs:**")
                        st.dataframe(cluster_programs[['title', 'financial_goals', 'age_bracket']].head(5))
        
        with tab2:
            st.markdown("### Program Rating Analysis")
            
            # Rating distribution
            rating_dist = rating_data['rating'].value_counts().sort_index().reset_index()
            rating_dist.columns = ['Rating', 'Count']
            
            fig = px.bar(
                rating_dist,
                x='Rating',
                y='Count',
                labels={'Rating': 'Rating Value', 'Count': 'Number of Ratings'},
                title="Overall Rating Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rating trends over time
            if 'timestamp' in rating_data.columns:
                rating_data['month'] = rating_data['timestamp'].dt.to_period('M')
                monthly_ratings = rating_data.groupby('month')['rating'].mean().reset_index()
                monthly_ratings['month'] = monthly_ratings['month'].astype(str)
                
                fig = px.line(
                    monthly_ratings,
                    x='month',
                    y='rating',
                    labels={'month': 'Month', 'rating': 'Average Rating'},
                    title="Rating Trends Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Ratings by financial goal
            st.markdown("### Ratings by Financial Goal")
            
            # Merge rating data with program info
            ratings_with_info = pd.merge(rating_data, users_with_finance[['title', 'financial_goals']], on='title')
            
            goal_ratings = ratings_with_info.groupby('financial_goals')['rating'].agg(['mean', 'count']).reset_index()
            goal_ratings = goal_ratings.sort_values('mean', ascending=False)
            
            fig = px.bar(
                goal_ratings,
                x='financial_goals',
                y='mean',
                labels={'financial_goals': 'Financial Goal', 'mean': 'Average Rating', 'count': 'Number of Ratings'},
                title="Average Rating by Financial Goal",
                hover_data=['count'],
                color='count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### User Demographics and Behavior")
            
            # User activity distribution
            user_activity = rating_data['userId'].value_counts().reset_index()
            user_activity.columns = ['userId', 'rating_count']
            
            fig = px.histogram(
                user_activity,
                x='rating_count',
                nbins=20,
                labels={'rating_count': 'Number of Ratings', 'count': 'Number of Users'},
                title="User Activity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Age bracket preferences
            ratings_with_age = pd.merge(rating_data, users_with_finance[['title', 'age_bracket']], on='title')
            user_age_prefs = ratings_with_age.groupby(['userId', 'age_bracket']).size().reset_index()
            user_age_prefs.columns = ['userId', 'age_bracket', 'count']
            
            # Find dominant age bracket per user
            dominant_age = user_age_prefs.loc[user_age_prefs.groupby('userId')['count'].idxmax()]
            age_dist = dominant_age['age_bracket'].value_counts().reset_index()
            age_dist.columns = ['age_bracket', 'user_count']
            
            fig = px.pie(
                age_dist,
                values='user_count',
                names='age_bracket',
                title="User Age Bracket Preferences"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Financial goals preferences
            ratings_with_goals = pd.merge(rating_data, users_with_finance[['title', 'financial_goals']], on='title')
            user_goal_prefs = ratings_with_goals.groupby(['userId', 'financial_goals']).size().reset_index()
            user_goal_prefs.columns = ['userId', 'financial_goals', 'count']
            
            # Find dominant financial goal per user
            dominant_goal = user_goal_prefs.loc[user_goal_prefs.groupby('userId')['count'].idxmax()]
            goal_dist = dominant_goal['financial_goals'].value_counts().reset_index()
            goal_dist.columns = ['financial_goals', 'user_count']
            
            fig = px.bar(
                goal_dist,
                x='financial_goals',
                y='user_count',
                labels={'financial_goals': 'Financial Goal', 'user_count': 'Number of Users'},
                title="User Financial Goal Preferences",
                color='user_count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
