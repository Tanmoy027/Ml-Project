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
warnings.filterwarnings('ignore')

class VulnerabilityRandomForest:
    def __init__(self, data_path="E:/project dataset/processed_data/processed_data"):
        """
        Initialize the Random Forest vulnerability detection model
        
        Args:
            data_path (str): Path to the processed data directory
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
            
            # Load the datasets
            self.train_df = pd.read_csv(os.path.join(self.data_path, 'train_features.csv'))
            self.val_df = pd.read_csv(os.path.join(self.data_path, 'val_features.csv'))
            self.test_df = pd.read_csv(os.path.join(self.data_path, 'test_features.csv'))
            
            print(f"Train set: {len(self.train_df)} samples")
            print(f"Validation set: {len(self.val_df)} samples") 
            print(f"Test set: {len(self.test_df)} samples")
            
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
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def preprocess_features(self, apply_smote=True, scale_features=True):
        """
        Preprocess features with optional SMOTE and scaling
        
        Args:
            apply_smote (bool): Whether to apply SMOTE for handling class imbalance
            scale_features (bool): Whether to scale features
        """
        print("\nPreprocessing features...")
        
        # Handle missing values
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # Apply SMOTE for handling class imbalance
        if apply_smote:
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Increase minority class to 30%
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"After SMOTE - Vulnerable samples: {(self.y_train_resampled == 1).sum()}")
            print(f"After SMOTE - Non-vulnerable samples: {(self.y_train_resampled == 0).sum()}")
        else:
            self.X_train_resampled = self.X_train.copy()
            self.y_train_resampled = self.y_train.copy()
        
        # Scale features
        if scale_features:
            print("Scaling features...")
            self.X_train_resampled = self.scaler.fit_transform(self.X_train_resampled)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)
        
    def train_random_forest(self, use_grid_search=True):
        """
        Train Random Forest model with hyperparameter tuning
        
        Args:
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
        """
        print("\nTraining Random Forest model...")
        
        if use_grid_search:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            # Create Random Forest classifier
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Grid search with cross-validation
            print("Performing grid search with 5-fold cross-validation...")
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', 
                n_jobs=-1, verbose=1, return_train_score=True
            )
            
            grid_search.fit(self.X_train_resampled, self.y_train_resampled)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters but balanced class weights
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(self.X_train_resampled, self.y_train_resampled)
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\nEvaluating model performance...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_resampled)
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)
        
        # Prediction probabilities
        y_train_proba = self.model.predict_proba(self.X_train_resampled)[:, 1]
        y_val_proba = self.model.predict_proba(self.X_val)[:, 1]
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics for each set
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
        
        # Detailed classification report for test set
        print(f"\nDetailed Test Set Classification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        return self.results
    
    def plot_results(self, save_path="./results"):
        """Generate comprehensive result visualizations"""
        print("\nGenerating visualizations...")
        
        # Create results directory
        os.makedirs(save_path, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Feature Importance
        plt.subplot(2, 3, 1)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        
        # 2. Confusion Matrix
        plt.subplot(2, 3, 2)
        cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Vulnerable', 'Vulnerable'],
                   yticklabels=['Non-Vulnerable', 'Vulnerable'])
        plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 3. ROC Curves
        plt.subplot(2, 3, 3)
        for name, (y_true, _, y_proba) in [
            ('Train', (self.y_train_resampled, None, self.model.predict_proba(self.X_train_resampled)[:, 1])),
            ('Validation', (self.y_val, None, self.model.predict_proba(self.X_val)[:, 1])),
            ('Test', (self.y_test, None, self.model.predict_proba(self.X_test)[:, 1]))
        ]:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curves
        plt.subplot(2, 3, 4)
        for name, (y_true, _, y_proba) in [
            ('Train', (self.y_train_resampled, None, self.model.predict_proba(self.X_train_resampled)[:, 1])),
            ('Validation', (self.y_val, None, self.model.predict_proba(self.X_val)[:, 1])),
            ('Test', (self.y_test, None, self.model.predict_proba(self.X_test)[:, 1]))
        ]:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            plt.plot(recall, precision, label=f'{name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Performance Metrics Comparison
        plt.subplot(2, 3, 5)
        metrics_df = pd.DataFrame(self.results).T
        metrics_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']].plot(kind='bar')
        plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 6. Class Distribution
        plt.subplot(2, 3, 6)
        class_dist = pd.DataFrame({
            'Original Train': [len(self.y_train) - self.y_train.sum(), self.y_train.sum()],
            'SMOTE Train': [len(self.y_train_resampled) - self.y_train_resampled.sum(), 
                           self.y_train_resampled.sum()],
            'Test': [len(self.y_test) - self.y_test.sum(), self.y_test.sum()]
        }, index=['Non-Vulnerable', 'Vulnerable'])
        
        class_dist.plot(kind='bar')
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=0)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'rf_vulnerability_detection_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance as CSV
        feature_importance.to_csv(os.path.join(save_path, 'feature_importance.csv'), index=False)
        print(f"Results saved to {save_path}")
    
    def cross_validate(self, cv_folds=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Different scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(self.model, self.X_train_resampled, self.y_train_resampled, 
                                   cv=cv, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            print(f"{metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_model(self, model_path="./models"):
        """Save the trained model and preprocessing objects"""
        print(f"\nSaving model to {model_path}...")
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, os.path.join(model_path, 'random_forest_vulnerability_model.pkl'))
        
        # Save the scaler
        joblib.dump(self.scaler, os.path.join(model_path, 'feature_scaler.pkl'))
        
        # Save feature names
        with open(os.path.join(model_path, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        # Save results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(os.path.join(model_path, 'model_performance.csv'))
        
        print("Model and artifacts saved successfully!")
    
    def run_complete_pipeline(self, apply_smote=True, use_grid_search=True, save_results=True):
        """Run the complete machine learning pipeline"""
        print("="*60)
        print("VULNERABILITY DETECTION - RANDOM FOREST PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Preprocess features
        self.preprocess_features(apply_smote=apply_smote)
        
        # Step 3: Train model
        self.train_random_forest(use_grid_search=use_grid_search)
        
        # Step 4: Evaluate model
        results = self.evaluate_model()
        
        # Step 5: Cross-validation
        cv_results = self.cross_validate()
        
        # Step 6: Generate visualizations
        if save_results:
            self.plot_results()
            self.save_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True

# Usage Example
if __name__ == "__main__":
    # Initialize the model
    rf_model = VulnerabilityRandomForest(
        data_path="E:/project dataset/processed_data/processed_data"
    )
    
    # Run the complete pipeline
    success = rf_model.run_complete_pipeline(
        apply_smote=True,      # Handle class imbalance
        use_grid_search=True,  # Hyperparameter tuning
        save_results=True      # Save model and results
    )
    
    if success:
        print("\nModel training completed successfully!")
        print("Check the './results' and './models' directories for outputs.")
    else:
        print("Model training failed. Please check the data paths and try again.")