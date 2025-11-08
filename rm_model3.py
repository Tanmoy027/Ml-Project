import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
import time
import psutil
import multiprocessing
import gc  # Garbage collector for memory management
warnings.filterwarnings('ignore')

class VulnerabilityRandomForest:
    def __init__(self, data_path="E:/project dataset/processed_data/processed_data"):
        """
        Initialize the Random Forest vulnerability detection model optimized for LOW MEMORY
        
        Args:
            data_path (str): Path to the processed data directory
        """
        self.data_path = data_path
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
        # Hardware optimization for LOW MEMORY (0.6GB available)
        self.n_cores = min(multiprocessing.cpu_count(), 2)  # Use only 2 cores to save memory
        self.memory_limit = psutil.virtual_memory().available / (1024**3)  # Available RAM in GB
        
        print(f"Hardware detected: {multiprocessing.cpu_count()} CPU cores, {self.memory_limit:.1f}GB available RAM")
        print("⚠️  LOW MEMORY DETECTED - Using memory-optimized settings")
        print("Note: Random Forest uses CPU only (no GPU acceleration available)")
        
    def estimate_training_time(self, n_samples, n_features, use_grid_search=True):
        """Estimate training time for LOW MEMORY setup"""
        print(f"\n{'='*50}")
        print("TIME ESTIMATION FOR LOW MEMORY SETUP")
        print(f"{'='*50}")
        
        # Adjust for memory constraints
        if self.memory_limit < 1.0:  # Less than 1GB available
            print("⚠️  CRITICAL: Very low memory detected!")
            print("Recommendations:")
            print("- Disable SMOTE (uses too much memory)")
            print("- Use smaller n_estimators (50-100)")
            print("- Reduce max_depth (5-10)")
        
        # Base time estimation (conservative for low memory)
        base_time_per_tree = 0.02  # Slower due to memory constraints
        
        if use_grid_search:
            estimated_time = 999999  # Discourage grid search
            print("❌ Grid Search NOT RECOMMENDED with low memory")
        else:
            n_estimators = 100  # Reduced from 200
            estimated_time = (n_estimators * base_time_per_tree * (n_samples / 1000)) / self.n_cores
            print(f"Single Model Training (Memory Optimized):")
            print(f"  - Estimators: {n_estimators} (reduced for memory)")
        
        print(f"\nDataset Info:")
        print(f"  - Training samples: {n_samples:,}")
        print(f"  - Features: {n_features}")
        print(f"  - CPU cores used: {self.n_cores} (reduced for memory)")
        
        # Convert to human readable time
        if estimated_time < 60:
            time_str = f"{estimated_time:.1f} seconds"
        elif estimated_time < 3600:
            time_str = f"{estimated_time/60:.1f} minutes"
        else:
            time_str = f"{estimated_time/3600:.1f} hours"
        
        print(f"\nEstimated Training Time: {time_str}")
        print(f"{'='*50}\n")
        
        return estimated_time
        
    def load_data(self):
        """Load data with memory optimization"""
        try:
            start_time = time.time()
            print("Loading preprocessed datasets with memory optimization...")
            
            # Load with dtype optimization to save memory
            dtype_dict = {
                'vul': 'int8',  # 0 or 1
                'Score': 'float32',  # Reduce from float64
                'code_length': 'int32',
                'token_count': 'int32',
                'has_malloc': 'int8',
                'has_pointers': 'int8',
                'add_lines': 'int32',
                'del_lines': 'int32'
            }
            
            # Load the datasets with optimized dtypes
            self.train_df = pd.read_csv(os.path.join(self.data_path, 'train_features.csv'), dtype=dtype_dict)
            self.val_df = pd.read_csv(os.path.join(self.data_path, 'val_features.csv'), dtype=dtype_dict)
            self.test_df = pd.read_csv(os.path.join(self.data_path, 'test_features.csv'), dtype=dtype_dict)
            
            load_time = time.time() - start_time
            print(f"Data loaded in {load_time:.2f} seconds")
            
            print(f"Train set: {len(self.train_df)} samples")
            print(f"Validation set: {len(self.val_df)} samples") 
            print(f"Test set: {len(self.test_df)} samples")
            
            # Check memory usage
            memory_usage = (self.train_df.memory_usage(deep=True).sum() + 
                          self.val_df.memory_usage(deep=True).sum() + 
                          self.test_df.memory_usage(deep=True).sum()) / (1024**2)
            print(f"Dataset memory usage: {memory_usage:.1f} MB")
            
            # Separate features and target
            feature_cols = [col for col in self.train_df.columns if col != 'vul']
            self.feature_names = feature_cols
            
            # Training data
            self.X_train = self.train_df[feature_cols]
            self.y_train = self.train_df['vul']
            
            # Validation data
            self.X_val = self.val_df[feature_cols]
            self.y_val = self.val_df['vul']
            
            # Test data
            self.X_test = self.test_df[feature_cols]
            self.y_test = self.test_df['vul']
            
            # Print class distribution
            print(f"\nClass distribution in training set:")
            print(f"Non-vulnerable: {(self.y_train == 0).sum()} ({(self.y_train == 0).mean()*100:.2f}%)")
            print(f"Vulnerable: {(self.y_train == 1).sum()} ({(self.y_train == 1).mean()*100:.2f}%)")
            
            # Estimate training time
            self.estimate_training_time(len(self.X_train), len(feature_cols), use_grid_search=False)
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_features(self, apply_smote=False, scale_features=True):
        """
        Preprocess features with MEMORY OPTIMIZATION
        SMOTE disabled by default due to memory constraints
        """
        start_time = time.time()
        print("\nPreprocessing features (memory optimized)...")
        
        # Handle missing values
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # Check available memory before SMOTE
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if apply_smote and available_memory > 2.0:  # Only if >2GB available
            print("Applying SMOTE for class imbalance...")
            # FIXED: Remove n_jobs parameter from SMOTE
            smote = SMOTE(random_state=42, sampling_strategy=0.15)  # Further reduced to 15%
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"After SMOTE - Vulnerable samples: {(self.y_train_resampled == 1).sum()}")
            print(f"After SMOTE - Non-vulnerable samples: {(self.y_train_resampled == 0).sum()}")
            
            # Force garbage collection after SMOTE
            gc.collect()
            
        elif apply_smote and available_memory <= 2.0:
            print("⚠️  SMOTE skipped due to insufficient memory")
            print("Using class_weight='balanced' instead for handling imbalance")
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
        else:
            print("SMOTE disabled - using original training set with balanced class weights")
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
        
        # Scale features with memory optimization
        if scale_features:
            print("Scaling features...")
            # Convert to numpy arrays to save memory
            self.X_train_resampled = self.scaler.fit_transform(self.X_train_resampled.values)
            self.X_val = self.scaler.transform(self.X_val.values)
            self.X_test = self.scaler.transform(self.X_test.values)
        
        # Force garbage collection
        gc.collect()
        
        preprocess_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        
    def train_random_forest(self, use_grid_search=False):
        """
        Train Random Forest model optimized for LOW MEMORY
        """
        start_time = time.time()
        print(f"\nTraining Random Forest model (memory optimized)...")
        
        if use_grid_search:
            print("❌ Grid search disabled due to memory constraints")
            use_grid_search = False
        
        # Memory-optimized Random Forest
        print("Training single optimized model...")
        self.model = RandomForestClassifier(
            n_estimators=100,        # Reduced from 200
            max_depth=15,            # Reduced from 20
            min_samples_split=10,    # Increased to reduce memory
            min_samples_leaf=5,      # Increased to reduce memory
            max_features='sqrt',
            class_weight='balanced',  # Handle imbalance without SMOTE
            random_state=42,
            n_jobs=self.n_cores,
            max_samples=0.7,         # Use only 70% of samples per tree
            bootstrap=True
        )
        
        self.model.fit(self.X_train_resampled, self.y_train_resampled)
        
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time/60:.1f} minutes")
        
        # Force garbage collection after training
        gc.collect()
    
    def evaluate_model(self):
        """Model evaluation with memory optimization"""
        start_time = time.time()
        print("\nEvaluating model performance...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_resampled)
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)
        
        # Prediction probabilities
        y_train_proba = self.model.predict_proba(self.X_train_resampled)[:, 1]
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        datasets = {
            'Train': (self.y_train_resampled, y_train_pred, y_train_proba),
            'Validation': (self.y_val, y_val_pred, y_val_proba),
            'Test': (self.y_test, y_test_pred, y_test_proba)
        }
        
        self.results = {}
        
        for name, (y_true, y_pred, y_proba) in datasets.items():
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': classification_report(y_true, y_pred, output_dict=True)['1']['precision'],
                'recall': classification_report(y_true, y_pred, output_dict=True)['1']['recall'],
                'f1': f1_score(y_true, y_pred),
                'auc_roc': roc_auc_score(y_true, y_proba)
            }
            
            self.results[name] = metrics
            
            print(f"\n{name} Set Performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print(f"\nDetailed Test Set Classification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        eval_time = time.time() - start_time
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        return self.results
    
    def plot_results(self, save_path="./results"):
        """Generate essential visualizations only (memory optimized)"""
        start_time = time.time()
        print("\nGenerating essential visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Create smaller figure to save memory
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Feature Importance
        plt.subplot(2, 2, 1)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance', fontsize=12, fontweight='bold')
        plt.xlabel('Importance Score')
        
        # 2. Confusion Matrix
        plt.subplot(2, 2, 2)
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Vulnerable', 'Vulnerable'],
                   yticklabels=['Non-Vulnerable', 'Vulnerable'])
        plt.title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 3. ROC Curve (Test set only to save memory)
        plt.subplot(2, 2, 3)
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        auc = roc_auc_score(self.y_test, y_test_proba)
        plt.plot(fpr, tpr, label=f'Test (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Performance Metrics
        plt.subplot(2, 2, 4)
        metrics_df = pd.DataFrame(self.results).T
        metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(kind='bar')
        plt.title('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'rf_results_memory_optimized.png'), 
                   dpi=100, bbox_inches='tight')  # Lower DPI to save memory
        plt.show()
        
        # Save feature importance
        feature_importance.to_csv(os.path.join(save_path, 'feature_importance.csv'), index=False)
        
        plot_time = time.time() - start_time
        print(f"Visualizations completed in {plot_time:.2f} seconds")
        print(f"Results saved to {save_path}")
    
    def save_model(self, model_path="./models"):
        """Save the trained model"""
        print(f"\nSaving model to {model_path}...")
        
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_path, 'rf_model_memory_optimized.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(os.path.join(model_path, 'performance.csv'))
        
        print("Model saved successfully!")
    
    def run_complete_pipeline(self, apply_smote=False, save_results=True):
        """Run the complete pipeline optimized for LOW MEMORY"""
        total_start_time = time.time()
        
        print("="*60)
        print("VULNERABILITY DETECTION - MEMORY OPTIMIZED PIPELINE")
        print("FOR LOW MEMORY SYSTEMS (< 1GB AVAILABLE)")
        print("="*60)
        
        if not self.load_data():
            return False
        
        self.preprocess_features(apply_smote=apply_smote)
        self.train_random_forest(use_grid_search=False)
        results = self.evaluate_model()
        
        if save_results:
            self.plot_results()
            self.save_model()
        
        total_time = time.time() - total_start_time
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETED IN {total_time/60:.1f} MINUTES!")
        print("="*60)
        
        return True

# Usage for LOW MEMORY systems
if __name__ == "__main__":
    rf_model = VulnerabilityRandomForest(
        data_path="E:/project dataset/processed_data/processed_data"
    )
    
    # Run with memory-optimized settings
    success = rf_model.run_complete_pipeline(
        apply_smote=False,       # Disabled due to memory constraints
        save_results=True
    )
    
    if success:
        print("\n✅ Model training completed successfully!")
        print("Check './results' and './models' directories for outputs.")
    else:
        print("❌ Model training failed.")