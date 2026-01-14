"""
Enhanced Text Matching Module
==============================
Advanced text similarity functions for identity resolution.

This module provides multiple text matching algorithms to handle:
- Case variations (e.g., "MylesBraithwaite" vs "myles braithwaite")
- Spacing differences (e.g., "Arthur H" vs "ArthurHabrial")
- Phonetic similarities (e.g., "Stephen" vs "Steven")
- Typos and minor variations

Author: Identity Resolution System
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import re


# ============================================================
# DEPENDENCY CHECK & INSTALLATION HELPER
# ============================================================

def check_dependencies():
    """Check if required libraries are installed."""
    missing = []
    
    try:
        import rapidfuzz
    except ImportError:
        missing.append('rapidfuzz')
    
    try:
        import jellyfish
    except ImportError:
        missing.append('jellyfish')
    
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


# ============================================================
# 1. FUZZY STRING MATCHING
# ============================================================

def fuzzy_ratio(s1: str, s2: str) -> float:
    """
    Calculate fuzzy string similarity using Levenshtein distance.
    
    Uses rapidfuzz for fast fuzzy matching.
    Returns value between 0.0 and 1.0.
    
    Examples:
        >>> fuzzy_ratio("MylesBraithwaite", "Myles Braithwaite")
        0.94
        >>> fuzzy_ratio("arthurhabrial", "Arthur Habrial")
        0.93
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    try:
        from rapidfuzz import fuzz
        # Use ratio for basic similarity
        score = fuzz.ratio(str(s1).lower(), str(s2).lower())
        return score / 100.0
    except ImportError:
        # Fallback to simple sequence matcher
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()


def fuzzy_token_sort_ratio(s1: str, s2: str) -> float:
    """
    Token-based fuzzy matching (order-independent).
    
    Handles cases where words are in different order:
    - "Myles Braithwaite" vs "Braithwaite Myles"
    
    Returns value between 0.0 and 1.0.
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    try:
        from rapidfuzz import fuzz
        score = fuzz.token_sort_ratio(str(s1).lower(), str(s2).lower())
        return score / 100.0
    except ImportError:
        # Fallback: just use regular ratio
        return fuzzy_ratio(s1, s2)


def fuzzy_partial_ratio(s1: str, s2: str) -> float:
    """
    Partial fuzzy matching for substring matches.
    
    Useful when one string is contained in another:
    - "Arthur H" vs "Arthur Habrial"
    - "David" vs "David Irigoin"
    
    Returns value between 0.0 and 1.0.
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    try:
        from rapidfuzz import fuzz
        score = fuzz.partial_ratio(str(s1).lower(), str(s2).lower())
        return score / 100.0
    except ImportError:
        # Fallback: check if one is substring of other
        s1_lower = str(s1).lower()
        s2_lower = str(s2).lower()
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 0.8
        return fuzzy_ratio(s1, s2)


# ============================================================
# 2. PHONETIC MATCHING
# ============================================================

def phonetic_similarity(s1: str, s2: str, algorithm: str = 'metaphone') -> float:
    """
    Calculate phonetic similarity (how similar two strings sound).
    
    Algorithms:
    - 'metaphone': General purpose phonetic algorithm
    - 'soundex': Classic phonetic algorithm (less accurate)
    
    Useful for:
    - Name variations: "Stephen" vs "Steven"
    - Spelling variations: "Catherine" vs "Katherine"
    
    Returns 1.0 if phonetically identical, 0.0 otherwise.
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    try:
        import jellyfish
        
        s1_str = str(s1).strip()
        s2_str = str(s2).strip()
        
        if algorithm == 'metaphone':
            # Metaphone is more accurate than Soundex
            code1 = jellyfish.metaphone(s1_str)
            code2 = jellyfish.metaphone(s2_str)
        elif algorithm == 'soundex':
            code1 = jellyfish.soundex(s1_str)
            code2 = jellyfish.soundex(s2_str)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Return 1.0 if codes match, 0.0 otherwise
        return 1.0 if code1 == code2 else 0.0
        
    except ImportError:
        # Fallback: return 0.0 if library not available
        return 0.0


# ============================================================
# 3. N-GRAM SIMILARITY
# ============================================================

def ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """
    Calculate n-gram similarity (character-level overlap).
    
    Good for:
    - Usernames with variations: "dvdrgn" vs "davidrgn"
    - Typos and minor spelling differences
    
    Args:
        n: Size of n-grams (default 2 = bigrams)
    
    Returns value between 0.0 and 1.0 (Jaccard similarity).
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    s1_lower = str(s1).lower()
    s2_lower = str(s2).lower()
    
    # Generate n-grams
    def get_ngrams(s, n):
        if len(s) < n:
            return {s}
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    
    ngrams1 = get_ngrams(s1_lower, n)
    ngrams2 = get_ngrams(s2_lower, n)
    
    # Calculate Jaccard similarity
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


# ============================================================
# 4. NORMALIZED EDIT DISTANCE
# ============================================================

def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein distance (edit distance).
    
    Calculates minimum number of edits (insertions, deletions, substitutions)
    needed to transform s1 into s2, normalized by string length.
    
    Returns value between 0.0 and 1.0 (1.0 = identical).
    """
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    
    if not s1 or not s2:
        return 0.0
    
    try:
        from rapidfuzz import distance
        s1_str = str(s1).lower()
        s2_str = str(s2).lower()
        
        # Calculate Levenshtein distance
        dist = distance.Levenshtein.distance(s1_str, s2_str)
        
        # Normalize by max length
        max_len = max(len(s1_str), len(s2_str))
        if max_len == 0:
            return 1.0
        
        # Convert distance to similarity (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - (dist / max_len)
        return max(0.0, similarity)
        
    except ImportError:
        # Fallback to basic implementation
        s1_str = str(s1).lower()
        s2_str = str(s2).lower()
        
        # Simple Levenshtein implementation
        if s1_str == s2_str:
            return 1.0
        
        len1, len2 = len(s1_str), len(s2_str)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Create distance matrix
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1_str[i-1] == s2_str[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        dist = matrix[len1][len2]
        max_len = max(len1, len2)
        return 1.0 - (dist / max_len)


# ============================================================
# 5. COMPOSITE MATCHING FUNCTIONS
# ============================================================

def enhanced_name_similarity(name1: str, name2: str) -> Tuple[float, dict]:
    """
    Enhanced name matching using multiple algorithms.
    
    Combines:
    - Fuzzy matching (handles case/spacing)
    - Token sort (handles word order)
    - Partial matching (handles initials/abbreviations)
    - Phonetic matching (handles spelling variations)
    
    Returns:
        Tuple of (final_score, component_scores_dict)
    """
    if pd.isna(name1) or pd.isna(name2):
        return 0.0, {}
    
    # Calculate all component scores
    scores = {
        'fuzzy': fuzzy_ratio(name1, name2),
        'token_sort': fuzzy_token_sort_ratio(name1, name2),
        'partial': fuzzy_partial_ratio(name1, name2),
        'phonetic': phonetic_similarity(name1, name2),
        'ngram': ngram_similarity(name1, name2, n=2)
    }
    
    # Weighted combination
    # Give more weight to fuzzy and token_sort as they're most reliable
    weights = {
        'fuzzy': 0.35,
        'token_sort': 0.30,
        'partial': 0.15,
        'phonetic': 0.10,
        'ngram': 0.10
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    
    return final_score, scores


def enhanced_username_similarity(user1: str, user2: str) -> Tuple[float, dict]:
    """
    Enhanced username matching.
    
    Usernames are typically:
    - Case-insensitive
    - No spaces
    - May have variations (@username vs username)
    
    Returns:
        Tuple of (final_score, component_scores_dict)
    """
    if pd.isna(user1) or pd.isna(user2):
        return 0.0, {}
    
    # Clean usernames (remove @ symbol, lowercase)
    clean1 = re.sub(r'^@', '', str(user1)).lower().strip()
    clean2 = re.sub(r'^@', '', str(user2)).lower().strip()
    
    # Calculate component scores
    scores = {
        'exact': 1.0 if clean1 == clean2 else 0.0,
        'fuzzy': fuzzy_ratio(clean1, clean2),
        'ngram': ngram_similarity(clean1, clean2, n=2),
        'edit_distance': normalized_levenshtein(clean1, clean2)
    }
    
    # Weighted combination
    # For usernames, exact match and edit distance are most important
    weights = {
        'exact': 0.40,
        'fuzzy': 0.30,
        'ngram': 0.15,
        'edit_distance': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    
    return final_score, scores


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def extract_name_components(full_name: str) -> dict:
    """
    Extract first name, last name, and initials from full name.
    
    Examples:
        >>> extract_name_components("Myles Braithwaite")
        {'first': 'Myles', 'last': 'Braithwaite', 'initials': 'MB'}
        >>> extract_name_components("Arthur H")
        {'first': 'Arthur', 'last': 'H', 'initials': 'AH'}
    """
    if pd.isna(full_name) or not full_name:
        return {'first': '', 'last': '', 'initials': ''}
    
    name_str = str(full_name).strip()
    parts = name_str.split()
    
    if len(parts) == 0:
        return {'first': '', 'last': '', 'initials': ''}
    elif len(parts) == 1:
        return {
            'first': parts[0],
            'last': '',
            'initials': parts[0][0].upper() if parts[0] else ''
        }
    else:
        first = parts[0]
        last = parts[-1]
        initials = ''.join([p[0].upper() for p in parts if p])
        
        return {
            'first': first,
            'last': last,
            'initials': initials
        }


def name_component_similarity(name1: str, name2: str) -> float:
    """
    Compare names by components (first, last, initials).
    
    Useful for cases like:
    - "Arthur H" vs "Arthur Habrial" (first name + initial)
    - "Myles Braithwaite" vs "M. Braithwaite" (initial + last name)
    """
    comp1 = extract_name_components(name1)
    comp2 = extract_name_components(name2)
    
    # Calculate component similarities
    first_sim = fuzzy_ratio(comp1['first'], comp2['first'])
    last_sim = fuzzy_ratio(comp1['last'], comp2['last'])
    
    # Check initial matching
    initial_match = 0.0
    if comp1['initials'] and comp2['initials']:
        # Check if one initial set is prefix of the other
        i1 = comp1['initials']
        i2 = comp2['initials']
        if i1.startswith(i2) or i2.startswith(i1):
            initial_match = 1.0
        elif i1 == i2:
            initial_match = 1.0
    
    # Weighted combination
    # If we have both first and last, use them
    if comp1['last'] and comp2['last']:
        return 0.5 * first_sim + 0.5 * last_sim
    # If only first name, use first + initials
    else:
        return 0.7 * first_sim + 0.3 * initial_match


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED TEXT MATCHING MODULE - DEMO")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Some features may not work without dependencies.")
        print("Install with: pip install rapidfuzz jellyfish")
    
    print("\n1. NAME MATCHING EXAMPLES")
    print("-" * 60)
    
    test_names = [
        ("Myles Braithwaite", "MylesBraithwaite"),
        ("Arthur Habrial", "Arthur H"),
        ("David Irigoin", "DavidIrigoin"),
        ("Alice Wong", "AliceWong"),
        ("Stephen Smith", "Steven Smith"),  # Phonetic test
    ]
    
    for name1, name2 in test_names:
        score, components = enhanced_name_similarity(name1, name2)
        print(f"\n'{name1}' vs '{name2}'")
        print(f"  Final Score: {score:.3f}")
        print(f"  Components: {components}")
    
    print("\n\n2. USERNAME MATCHING EXAMPLES")
    print("-" * 60)
    
    test_usernames = [
        ("@mylesb", "myles"),
        ("arthurhabrial", "ArthurHabrial"),
        ("dvdrgn", "DavidIrigoin"),
        ("@Alice7vong", "alice7vong"),
    ]
    
    for user1, user2 in test_usernames:
        score, components = enhanced_username_similarity(user1, user2)
        print(f"\n'{user1}' vs '{user2}'")
        print(f"  Final Score: {score:.3f}")
        print(f"  Components: {components}")
    
    print("\n\n3. NAME COMPONENT EXTRACTION")
    print("-" * 60)
    
    test_full_names = [
        "Myles Braithwaite",
        "Arthur H",
        "David Irigoin",
        "Alice Wong"
    ]
    
    for name in test_full_names:
        components = extract_name_components(name)
        print(f"\n'{name}'")
        print(f"  Components: {components}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
