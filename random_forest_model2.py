import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
import time
warnings.filterwarnings('ignore')

class FastVulnerabilityRandomForest:
    def __init__(self, data_path="E:/project dataset/processed_data/processed_data"):
        """
        Optimized Random Forest for your hardware specs
        """
        self.data_path = data_path
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load the preprocessed training, validation, and test datasets"""
        try:
            print("Loading preprocessed datasets...")
            
            self.train_df = pd.read_csv(os.path.join(self.data_path, 'train_features.csv'))
            self.val_df = pd.read_csv(os.path.join(self.data_path, 'val_features.csv'))
            self.test_df = pd.read_csv(os.path.join(self.data_path, 'test_features.csv'))
            
            print(f"Train set: {len(self.train_df)} samples")
            print(f"Validation set: {len(self.val_df)} samples") 
            print(f"Test set: {len(self.test_df)} samples")
            
            # Separate features and target
            feature_cols = [col for col in self.train_df.columns if col != 'vul']
            self.feature_names = feature_cols
            
            self.X_train = self.train_df[feature_cols]
            self.y_train = self.train_df['vul']
            self.X_val = self.val_df[feature_cols]
            self.y_val = self.val_df['vul']
            self.X_test = self.test_df[feature_cols]
            self.y_test = self.test_df['vul']
            
            print(f"Vulnerable samples in training: {self.y_train.sum()} ({self.y_train.mean()*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_features(self, apply_smote=True, sampling_strategy=0.2):
        """
        Optimized preprocessing with reduced SMOTE ratio
        """
        print("\nPreprocessing features...")
        
        # Handle missing values
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        if apply_smote:
            print(f"Applying SMOTE with sampling_strategy={sampling_strategy}...")
            start_time = time.time()
            
            # SMOTE doesn't have n_jobs parameter - FIXED!
            smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
            
            smote_time = time.time() - start_time
            print(f"SMOTE completed in {smote_time:.1f} seconds")
            print(f"After SMOTE - Vulnerable: {(self.y_train_resampled == 1).sum()}")
            print(f"After SMOTE - Non-vulnerable: {(self.y_train_resampled == 0).sum()}")
        else:
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
    
    def train_baseline_model(self):
        """
        Train a baseline Random Forest with good default parameters
        This should complete in ~5-10 minutes
        """
        print("\nTraining Baseline Random Forest...")
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=100,        # Reduced from 200
            max_depth=15,           # Limited depth
            min_samples_split=10,   # Higher to prevent overfitting
            min_samples_leaf=5,     # Higher to prevent overfitting
            max_features='sqrt',    # Optimal for most cases
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,              # Use all cores
            verbose=1               # Show progress
        )
        
        self.model.fit(self.X_train_resampled, self.y_train_resampled)
        
        training_time = time.time() - start_time
        print(f"Baseline model trained in {training_time:.1f} seconds")
    
    def train_optimized_grid_search(self):
        """
        Much smaller grid search - should complete in 10-15 minutes
        """
        print("\nTraining with Optimized Grid Search...")
        start_time = time.time()
        
        # Smaller, focused parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],          # 3 options
            'max_depth': [10, 15, 20],               # 3 options  
            'min_samples_split': [5, 10],            # 2 options
            'max_features': ['sqrt', 'log2'],        # 2 options
            'class_weight': ['balanced']             # 1 option
        }
        
        # Total combinations: 3 × 3 × 2 × 2 × 1 = 36 (much better than 3240!)
        total_fits = np.prod([len(v) for v in param_grid.values()]) * 3  # 3-fold CV
        print(f"Testing {total_fits} model combinations (3-fold CV)")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=3,                    # Reduced from 5-fold to 3-fold
            scoring='f1', 
            n_jobs=2,                # Limited parallel jobs for your CPU
            verbose=2,               # Show progress
            return_train_score=False # Faster
        )
        
        grid_search.fit(self.X_train_resampled, self.y_train_resampled)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        training_time = time.time() - start_time
        print(f"Grid search completed in {training_time:.1f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self):
        """Quick evaluation"""
        print("\nEvaluating model...")
        
        # Predictions
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)
        
        # Probabilities
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        datasets = {
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
            
            print(f"\n{name} Performance:")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.4f}")
        
        return self.results
    
    def plot_quick_results(self):
        """Generate essential plots only"""
        print("\nGenerating key visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[0,0].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[0,0].set_yticks(range(len(feature_importance)))
        axes[0,0].set_yticklabels(feature_importance['feature'])
        axes[0,0].set_title('Feature Importance')
        axes[0,0].set_xlabel('Importance Score')
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title('Test Set Confusion Matrix')
        axes[0,1].set_ylabel('Actual')
        axes[0,1].set_xlabel('Predicted')
        
        # 3. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        auc = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        axes[1,0].plot(fpr, tpr, label=f'Test AUC = {auc:.3f}')
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Performance Comparison
        metrics_df = pd.DataFrame(self.results).T
        metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Performance Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].tick_params(axis='x', rotation=0)
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('fast_rf_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Results saved as 'fast_rf_results.png'")
    
    def save_model(self):
        """Save the model"""
        joblib.dump(self.model, 'fast_rf_vulnerability_model.pkl')
        
        # Also save results summary
        with open('model_summary.txt', 'w') as f:
            f.write("Random Forest Vulnerability Detection - Results Summary\n")
            f.write("="*50 + "\n\n")
            
            if hasattr(self, 'best_params') and self.best_params:
                f.write(f"Best Parameters: {self.best_params}\n\n")
            
            f.write("Performance Results:\n")
            for dataset, metrics in self.results.items():
                f.write(f"\n{dataset} Set:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        print("Model saved as 'fast_rf_vulnerability_model.pkl'")
        print("Results summary saved as 'model_summary.txt'")
    
    def run_fast_pipeline(self, use_grid_search=False):
        """
        Run optimized pipeline for your hardware
        
        Args:
            use_grid_search (bool): If False, uses baseline model (faster)
                                   If True, uses optimized grid search
        """
        total_start = time.time()
        
        print("="*60)
        print("FAST VULNERABILITY DETECTION - RANDOM FOREST")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Preprocess (reduced SMOTE ratio for speed)
        self.preprocess_features(apply_smote=True, sampling_strategy=0.2)
        
        # Train model
        if use_grid_search:
            self.train_optimized_grid_search()  # ~10-15 minutes
        else:
            self.train_baseline_model()         # ~5-10 minutes
        
        # Evaluate
        self.evaluate_model()
        
        # Plot results
        self.plot_quick_results()
        
        # Save model
        self.save_model()
        
        total_time = time.time() - total_start
        print(f"\n" + "="*60)
        print(f"PIPELINE COMPLETED IN {total_time/60:.1f} MINUTES!")
        print("="*60)
        
        return True

# Usage
if __name__ == "__main__":
    # Fixed path to match your directory structure
    rf_model = FastVulnerabilityRandomForest(
        data_path="E:\project dataset\processed_data\processed_data"  # Removed the duplicate folder name
    )
    
    # Choose one:
    # Option 1: Baseline model (fastest - ~5-10 minutes)
    rf_model.run_fast_pipeline(use_grid_search=False)
    
    # Option 2: Optimized grid search (~10-15 minutes)
    # rf_model.run_fast_pipeline(use_grid_search=True)