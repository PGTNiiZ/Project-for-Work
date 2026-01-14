import pandas as pd
from match_profile import calculate_similarity, decide_match
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_data():
    """Load profiles and pairs data."""
    # Load all profiles with embeddings
    profiles_df = pd.read_csv("/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/profiles_with_embeddings.csv")
    
    # Load ground truth pairs (Instagram - Google+)
    pairs_df = pd.read_csv("/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/pairs_instagram_googleplus.csv")
    
    return profiles_df, pairs_df

def evaluate(profiles_df, pairs_df):
    """Run evaluation on the pairs."""
    
    # Filter profiles for faster lookup
    # We need to look up by platform and userName (or profile_id if available in pairs used)
    # The pairs file has 'instagram_userName' and 'googleplus_userName'
    
    # Create lookup dictionaries for efficiency
    # Key: (platform, userName) -> Value: row dict
    
    profile_lookup = {}
    for _, row in profiles_df.iterrows():
        # Clean username to match pairs format if needed (simple lower case for now)
        key = (row['platform'], str(row['userName']))
        profile_lookup[key] = row.to_dict()

    y_true = []
    y_scores = []
    
    results = []

    print(f"Evaluating {len(pairs_df)} pairs...")
    
    found_count = 0
    missing_count = 0

    # Determine platform columns dynamically
    # Expecting columns like "{platform}_userName"
    cols = pairs_df.columns
    platform_cols = [c for c in cols if c.endswith('_userName') and not c.endswith('_clean')]
    
    if len(platform_cols) != 2:
        print(f"Error: Could not identify two platform columns. Found: {platform_cols}")
        return

    plat1 = platform_cols[0].replace('_userName', '')
    plat2 = platform_cols[1].replace('_userName', '')
    
    print(f"Evaluating match between {plat1} and {plat2}")

    for _, row in pairs_df.iterrows():
        # Get profiles
        p1_key = (plat1, str(row[f'{plat1}_userName']))
        p2_key = (plat2, str(row[f'{plat2}_userName']))
        
        # Check if we have both profiles
        if p1_key in profile_lookup and p2_key in profile_lookup:
            p1 = profile_lookup[p1_key]
            p2 = profile_lookup[p2_key]
            
            # Run matching
            sim_result = calculate_similarity(p1, p2)
            score = sim_result['confidence_score']
            
            y_true.append(row['is_match'])
            y_scores.append(score)
            
            found_count += 1
            
            # Store detail for debugging/analysis
            if len(results) < 5: # Store first 5 for sample
                results.append({
                    'p1': p1['userName'],
                    'p2': p2['userName'],
                    'score': score,
                    'is_match': row['is_match'],
                    'details': sim_result
                })
        else:
            missing_count += 1

    print(f"Pairs processed: {found_count}")
    print(f"Pairs missing profiles: {missing_count}")
    
    if not y_true:
        print("No pairs found to evaluate.")
        return

    # Calculate metrics at different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    print("\n--- Evaluation Results ---")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    
    for thresh in thresholds:
        y_pred = [1 if s >= thresh else 0 for s in y_scores]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"{thresh:<10} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

    print("\n--- Sample Results ---")
    for r in results:
        print(f"{r['p1']} <-> {r['p2']} | Score: {r['score']} | details: {r['details']}")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Evaluate matching algorithm.')
    parser.add_argument('pairs_file', nargs='?', help='Path to pairs CSV file')
    args = parser.parse_args()

    profiles_df = pd.read_csv("/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/profiles_with_embeddings.csv")
    
    pairs_path = args.pairs_file
    if not pairs_path:
        # Default
        pairs_path = "/Users/tm/Documents/KMITL/Senior_Project/Project-for-Work/data/processed/pairs_instagram_googleplus.csv"
    
    print(f"Loading pairs from: {pairs_path}")
    pairs_df = pd.read_csv(pairs_path)
    
    evaluate(profiles_df, pairs_df)
