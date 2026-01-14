import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

df = pd.read_csv("/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/profiles_with_embeddings.csv")
df.head()

def string_similarity(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def location_similarity(loc1, loc2):
    # Use string similarity for location (fuzzy match)
    return string_similarity(loc1, loc2)

def calculate_similarity(profile_a, profile_b):
    """
    Calculate similarity between two profiles with dynamic weighting.
    """
    # Calculate raw similarities
    name_sim = string_similarity(profile_a.get("fullName"), profile_b.get("fullName"))
    username_sim = string_similarity(profile_a.get("userName"), profile_b.get("userName"))
    location_sim = location_similarity(profile_a.get("location"), profile_b.get("location"))
    
    # Handle embedding similarity
    path_a = profile_a.get("embedding_path")
    path_b = profile_b.get("embedding_path")
    
    # Check availability for dynamic weighting
    # We consider a field "available" if it is not NaN.
    # For embedding, we check if path is not NaN.
    
    has_name = not pd.isna(profile_a.get("fullName")) and not pd.isna(profile_b.get("fullName"))
    has_username = not pd.isna(profile_a.get("userName")) and not pd.isna(profile_b.get("userName"))
    
    # For location, we might want to be stricter? No, if it's there, compare it.
    has_location = not pd.isna(profile_a.get("location")) and not pd.isna(profile_b.get("location"))
    
    has_embedding = False
    embed_sim = 0.0
    if not pd.isna(path_a) and not pd.isna(path_b):
        try:
            embed_sim = embedding_similarity(path_a, path_b)
            # If embedding_similarity returns 0.0 but didn't crash, we effectively count it.
            # But wait, if it returns 0.0 because of error (caught inside), we shouldn't count it?
            # The current implementation returns 0.0 on error or NaN.
            # Let's assume if paths exist, we try.
            has_embedding = True
        except:
            has_embedding = False

    # Define base weights
    weights = {
        "name": 0.3,
        "username": 0.3,
        "location": 0.2,
        "embedding": 0.2
    }
    
    # Filter active weights
    active_components = []
    if has_name: active_components.append("name")
    if has_username: active_components.append("username")
    if has_location: active_components.append("location")
    if has_embedding: active_components.append("embedding")
    
    # If no common data (shouldn't happen for valid pairs), return 0
    if not active_components:
        return {
            "name_similarity": 0.0,
            "username_similarity": 0.0,
            "location_similarity": 0.0,
            "embedding_similarity": 0.0,
            "confidence_score": 0.0
        }

    # Normalize weights
    total_weight = sum(weights[k] for k in active_components)
    normalized_weights = {k: weights[k]/total_weight for k in active_components}
    
    # Calculate score
    score = 0.0
    if "name" in active_components: score += name_sim * normalized_weights["name"]
    if "username" in active_components: score += username_sim * normalized_weights["username"]
    if "location" in active_components: score += location_sim * normalized_weights["location"]
    if "embedding" in active_components: score += embed_sim * normalized_weights["embedding"]
    
    return {
        "name_similarity": round(name_sim, 3),
        "username_similarity": round(username_sim, 3),
        "location_similarity": round(location_sim, 3),
        "embedding_similarity": round(embed_sim, 3),
        "confidence_score": round(score, 3)
    }

def decide_match(confidence, threshold_match=0.85, threshold_possible=0.6):
    if confidence >= threshold_match:
        return "Match"
    elif confidence >= threshold_possible:
        return "Possible Match"
    else:
        return "Not Match"

if __name__ == "__main__":
    # Example usage
    try:
        df = pd.read_csv("/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/profiles_with_embeddings.csv")
        
        googleplus_profile = df[(df["platform"] == "googleplus")].iloc[0]
        instagram_profile = df[(df["platform"] == "instagram")].iloc[0] # Just picking one for demo

        result = calculate_similarity(googleplus_profile, instagram_profile)
        result["decision"] = decide_match(result["confidence_score"])
        
        print("Match Result:", result)
    except Exception as e:
        print(f"Error running example: {e}")

