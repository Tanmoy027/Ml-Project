import pandas as pd
import numpy as np
import os
import re
import hashlib
from sklearn.model_selection import train_test_split
import pickle

def preprocess_vulnerability_dataset(input_file_path, output_dir="."):
    """
    Main preprocessing function following the paper's approach
    """
    print(f"Starting preprocessing pipeline for vulnerability detection...")
    
    # 1. Load the dataset
    print(f"Loading dataset from: {input_file_path}")
    df = pd.read_csv(input_file_path, low_memory=False)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Remove unnecessary index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # 2. Create a lexer for tokenizing code
    c_cpp_tokens = {
        # Keywords
        'if': 'KEYWORD_IF', 'else': 'KEYWORD_ELSE', 'for': 'KEYWORD_FOR', 
        'while': 'KEYWORD_WHILE', 'return': 'KEYWORD_RETURN',
        'int': 'TYPE_INT', 'char': 'TYPE_CHAR', 'void': 'TYPE_VOID',
        'float': 'TYPE_FLOAT', 'double': 'TYPE_DOUBLE',
        # Operators
        '+': 'OP_PLUS', '-': 'OP_MINUS', '*': 'OP_MULTIPLY', '/': 'OP_DIVIDE',
        '=': 'OP_ASSIGN', '==': 'OP_EQUALS', '!=': 'OP_NOT_EQUALS',
        '<': 'OP_LESS', '>': 'OP_GREATER', '<=': 'OP_LESS_EQUALS', 
        '>=': 'OP_GREATER_EQUALS', '&&': 'OP_AND', '||': 'OP_OR'
    }
    
    # 3. Define tokenization function
    def tokenize_code(code):
        """Tokenize source code to reduce vocabulary size"""
        if pd.isna(code) or not isinstance(code, str):
            return []
        
        # Remove comments
        code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE|re.DOTALL)
        
        # Define token pattern
        pattern = r'[a-zA-Z_]\w*|".*?"|\'.*?\'|\d+|==|!=|<=|>=|&&|\|\||[+\-*/=<>!&|^%(){}\[\].,;:]'
        tokens = []
        
        for match in re.finditer(pattern, code):
            token = match.group(0)
            
            # Map to predefined token types
            if token in c_cpp_tokens:
                tokens.append(c_cpp_tokens[token])
            elif re.match(r'^[0-9]+$', token):
                tokens.append('NUMBER')
            elif re.match(r'^".*"$|^\'.*\'$', token):
                tokens.append('STRING_LITERAL')
            elif re.match(r'^[a-zA-Z_]\w*$', token):
                # Function or variable name
                tokens.append('IDENTIFIER')
            else:
                tokens.append('OTHER')
        
        return tokens
    
    # 4. Extract and tokenize code from the dataset
    print("Tokenizing code...")
    df['code_tokens'] = df['func_before'].apply(tokenize_code)
    print(f"Code tokenization complete. Average tokens per function: {df['code_tokens'].apply(len).mean():.2f}")
    
    # 5. Generate hash for duplicate detection
    print("Removing duplicates...")
    df['code_hash'] = df['code_tokens'].apply(
        lambda tokens: hashlib.md5(" ".join(str(t) for t in tokens).encode()).hexdigest()
    )
    original_count = len(df)
    df = df.drop_duplicates(subset=['code_hash'])
    print(f"Removed {original_count - len(df)} duplicates. Remaining samples: {len(df)}")
    
    # 6. Extract features
    print("Extracting features...")
    # Create basic code complexity features
    df['code_length'] = df['func_before'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
    df['token_count'] = df['code_tokens'].apply(len)
    
    # Look for risky patterns (memory operations, pointer usage)
    df['has_malloc'] = df['func_before'].apply(
        lambda x: 1 if isinstance(x, str) and 'malloc' in x else 0
    )
    df['has_pointers'] = df['func_before'].apply(
        lambda x: 1 if isinstance(x, str) and '*' in x else 0
    )
    
    # Select features for the model
    feature_columns = [
        'vul',                  # Target variable
        'Score',                # Severity score
        'code_length',          # Code size
        'token_count',          # Complexity measure
        'has_malloc',           # Memory allocation flag
        'has_pointers',         # Pointer usage flag
        'add_lines',            # Lines added in commit
        'del_lines',            # Lines deleted in commit
    ]
    
    # Handle missing values in features
    features_df = df[feature_columns].copy()
    for col in features_df.columns:
        if features_df[col].dtype in [np.float64, np.int64]:
            features_df[col] = features_df[col].fillna(0)
    
    print(f"Feature extraction complete with {len(feature_columns)} features")
    
    # 7. Split dataset
    print("Creating train/validation/test splits...")
    X = features_df.drop('vul', axis=1)
    y = features_df['vul']
    
    # Stratified split to maintain vulnerable/non-vulnerable ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 8. Save processed datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save features as CSV
    train_df.to_csv(os.path.join(output_dir, 'processed_train_features.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'processed_val_features.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'processed_test_features.csv'), index=False)
    
    # Save tokenized code
    train_indices = X_train.index
    val_indices = X_val.index
    test_indices = X_test.index
    
    train_tokens = df.loc[train_indices, ['code_tokens', 'vul']]
    val_tokens = df.loc[val_indices, ['code_tokens', 'vul']]
    test_tokens = df.loc[test_indices, ['code_tokens', 'vul']]
    
    # Save tokenized code using pickle
    with open(os.path.join(output_dir, 'train_code_tokens.pkl'), 'wb') as f:
        pickle.dump(train_tokens, f)
    with open(os.path.join(output_dir, 'val_code_tokens.pkl'), 'wb') as f:
        pickle.dump(val_tokens, f)
    with open(os.path.join(output_dir, 'test_code_tokens.pkl'), 'wb') as f:
        pickle.dump(test_tokens, f)
    
    # 9. Create summary file
    print("Creating preprocessing summary...")
    with open(os.path.join(output_dir, 'preprocessing_summary.txt'), 'w') as f:
        f.write("Vulnerability Detection Dataset Preprocessing Summary\n")
        f.write("================================================\n\n")
        f.write(f"Original dataset: {original_count} samples\n")
        f.write(f"After duplicate removal: {len(df)} samples\n")
        f.write(f"Vulnerability distribution: {y.mean()*100:.2f}% vulnerable\n\n")
        f.write(f"Final dataset splits:\n")
        f.write(f"- Training: {len(train_df)} samples ({train_df['vul'].sum()} vulnerable)\n")
        f.write(f"- Validation: {len(val_df)} samples ({val_df['vul'].sum()} vulnerable)\n")
        f.write(f"- Testing: {len(test_df)} samples ({test_df['vul'].sum()} vulnerable)\n\n")
        f.write(f"Features extracted: {', '.join(feature_columns[1:])}\n")
    
    print("Preprocessing complete! Files saved to output directory.")
    print(f"Final dataset stats:")
    print(f"  - Train set: {len(train_df)} samples with {train_df['vul'].sum()} vulnerable")
    print(f"  - Val set: {len(val_df)} samples with {val_df['vul'].sum()} vulnerable")
    print(f"  - Test set: {len(test_df)} samples with {test_df['vul'].sum()} vulnerable")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens
    }

# Run the preprocessing pipeline
if __name__ == "__main__":
    input_path = r"E:\project dataset\git hub\MSR_data_cleaned\MSR_data_cleaned.csv"
    output_dir = r"E:\project dataset\processed_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run preprocessing
    preprocess_vulnerability_dataset(input_path, output_dir)