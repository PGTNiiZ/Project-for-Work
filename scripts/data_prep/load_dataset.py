"""
Script to load and combine all social media profiles from Dataset-LinkSocial
Combines data from 1.profile.data, 2.profile.data, and 3.profile.data folders
"""
import os
import json
import pandas as pd

def load_all_profiles(base_path="data/data/Dataset-LinkSocial"):
    """
    Load all profiles from all 3 profile.data folders and combine them.
    
    Returns:
        pd.DataFrame: Combined DataFrame with all profiles
    """
    all_profiles = []
    profile_folders = ["1.profile.data", "2.profile.data", "3.profile.data"]
    
    for folder in profile_folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
            
        print(f"Loading from {folder}...")
        
        # Iterate through each user folder
        for user_folder in os.listdir(folder_path):
            user_path = os.path.join(folder_path, user_folder)
            
            if os.path.isdir(user_path):
                # Read each JSON file in user folder
                for file in os.listdir(user_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(user_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                # Add metadata
                                data['source_folder'] = folder
                                data['user_folder'] = user_folder
                                
                                # Determine platform from filename
                                if 'twitter' in file.lower():
                                    data['platform'] = 'twitter'
                                elif 'instagram' in file.lower():
                                    data['platform'] = 'instagram'
                                elif 'google' in file.lower():
                                    data['platform'] = 'googleplus'
                                else:
                                    data['platform'] = 'unknown'
                                    
                                all_profiles.append(data)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_profiles)
    print(f"\nTotal profiles loaded: {len(df)}")
    print(f"Platforms: {df['platform'].value_counts().to_dict()}")
    print(f"Source folders: {df['source_folder'].value_counts().to_dict()}")
    
    return df


if __name__ == "__main__":
    # Load all profiles
    df = load_all_profiles()
    
    # Show sample
    print("\nSample data:")
    print(df.head())
    
    # Save to CSV
    output_path = "data/combined_profiles.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved combined dataset to: {output_path}")
