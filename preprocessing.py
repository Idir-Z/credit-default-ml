import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Reusable function to train a model and evaluate its performance.
    
    Parameters:
    -----------
    model : sklearn pipeline or estimator
        The model to train
    X_train, X_test : array-like
        Training and testing features
    y_train, y_test : array-like
        Training and testing labels
    model_name : str
        Name for display purposes
    
    Returns:
    --------
    dict : Dictionary containing predictions, probabilities, and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc
    }


def detect_leaky_features(X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, 
                          numeric_pipeline, categorical_pipeline, top_n=10):
    """
    Detect potential data leakage by training individual models on each feature.
    Features with suspiciously high AUC scores are likely leaky.
    
    Parameters:
    -----------
    X_train, X_test : DataFrame
        Training and testing features
    y_train, y_test : Series
        Training and testing labels
    numeric_cols, categorical_cols : list
        Lists of numeric and categorical column names
    numeric_pipeline, categorical_pipeline : Pipeline
        Preprocessing pipelines for numeric and categorical features
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    dict : Dictionary mapping feature names to their ROC-AUC scores
    """
    print(f"\n{'='*60}")
    print("Detecting Potential Data Leakage")
    print(f"{'='*60}")
    
    leakage_scores = {}
    
    for col in X_train.columns:
        # Determine which pipeline to use based on column type
        if col in numeric_cols:
            transformer = ('num', numeric_pipeline, [col])
        elif col in categorical_cols:
            transformer = ('cat', categorical_pipeline, [col])
        else:
            continue
        
        # Create a pipeline for this single feature
        model = Pipeline(steps=[
            ('prep', ColumnTransformer([transformer])),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        try:
            # Train on single feature
            model.fit(X_train[[col]], y_train)
            y_prob = model.predict_proba(X_test[[col]])[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            leakage_scores[col] = auc
        except Exception as e:
            print(f"Warning: Could not process {col}: {str(e)}")
            continue
    
    # Sort by ROC-AUC score (descending)
    sorted_scores = sorted(leakage_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop features by individual ROC-AUC (potential leakage indicators):")
    print(f"{'Feature':<30} {'ROC-AUC':<10}")
    print("-" * 40)
    for feat, score in sorted_scores[:top_n]:
        print(f"{feat:<30} {score:.4f}")
    
    return dict(sorted_scores)


def check_outliers(X_train_scaled, feature_names):
    """
    Check for outliers using Z-score method after scaling.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Scaled training features
    feature_names : list
        Names of features
    
    Returns:
    --------
    DataFrame : Number of outliers per feature
    """
    z_scores = np.abs(stats.zscore(X_train_scaled))
    outlier_mask = (z_scores > 3)
    num_outliers_per_feature = outlier_mask.sum(axis=0)
    
    print("\n" + "="*60)
    print("Outlier Detection (Z-score > 3)")
    print("="*60)
    
    outlier_df = pd.DataFrame({
        'Feature': feature_names,
        'Num_Outliers': num_outliers_per_feature
    })
    outlier_df = outlier_df[outlier_df['Num_Outliers'] > 0].sort_values('Num_Outliers', ascending=False)
    
    print(outlier_df)
    return outlier_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# Load data
df = pd.read_csv("Loan_Default.csv")

# Prepare features and target
# Missing data handling (leakage-safe) - drop ID and year as they're not predictive
X = df.drop(columns=['Status', 'ID', 'year'])
y = df['Status']

# Train/test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Separate numerical and categorical features
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"Number of numeric features: {len(numeric_cols)}")
print(f"Number of categorical features: {len(categorical_cols)}")

# ============================================================================
# PREPROCESSING PIPELINES
# ============================================================================

# Numerical pipeline: impute median, transform for normality, scale
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute most frequent, one-hot encode
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# ============================================================================
# OUTLIER ANALYSIS (DEMONSTRATION)
# ============================================================================

# Copy numeric columns for outlier analysis
X_train_num = X_train[numeric_cols].copy()
X_test_num = X_test[numeric_cols].copy()

# Apply transformations step by step for analysis
pt = PowerTransformer(method='yeo-johnson', standardize=False)
X_train_num_trans = pd.DataFrame(
    pt.fit_transform(X_train_num.fillna(X_train_num.median())), 
    columns=numeric_cols
)

scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_num_trans), 
    columns=numeric_cols
)

# Check outliers after transformation and scaling
check_outliers(X_train_num_scaled, numeric_cols)

# ============================================================================
# BASELINE ACCURACY
# ============================================================================

class_ratios = y_train.value_counts(normalize=True)
baseline_accuracy = class_ratios.max()
print(f"\n{'='*60}")
print(f"Baseline accuracy (majority class): {baseline_accuracy:.2%}")
print(f"{'='*60}")

# ============================================================================
# DATA LEAKAGE DETECTION
# ============================================================================

# Detect leaky features by training models on individual features
leakage_scores = detect_leaky_features(
    X_train, X_test, y_train, y_test, 
    numeric_cols, categorical_cols,
    numeric_pipeline, categorical_pipeline,
    top_n=10
)

# Identify leaky features (typically AUC > 0.9 on single feature is suspicious)
# These features are most likely derived after the default/non-default decision
# and would not be available at prediction time in production
leaky_features = [
    'Interest_rate_spread',
    'rate_of_interest',
    'Upfront_charges',
]

print(f"\n{'='*60}")
print(f"Removing leaky features: {leaky_features}")
print(f"{'='*60}")

# ============================================================================
# MODEL TRAINING WITH CLEANED FEATURES
# ============================================================================

# Remove leaky features from training and test sets
X_train_clean = X_train.drop(columns=leaky_features)
X_test_clean = X_test.drop(columns=leaky_features)

# Update column lists after removing leaky features
numeric_cols_clean = [col for col in numeric_cols if col not in leaky_features]
categorical_cols_clean = [col for col in categorical_cols if col not in leaky_features]

# Rebuild preprocessor with clean features
preprocessor_clean = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols_clean),
    ('cat', categorical_pipeline, categorical_cols_clean)
])

# Create clean baseline model
baseline_model_clean = Pipeline(steps=[
    ('preprocessor', preprocessor_clean),
    ('classifier', LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,
        random_state=42
    ))
])

# Train and evaluate the clean model
results = train_and_evaluate_model(
    baseline_model_clean, 
    X_train_clean, 
    X_test_clean, 
    y_train, 
    y_test,
    model_name="Baseline Logistic Regression (No Leaky Features)"
)

print(f"\n{'='*60}")
print("Pipeline Complete!")
print(f"{'='*60}")
print(f"Final ROC-AUC: {results['roc_auc']:.4f}")
print(f"This is a more realistic score compared to perfect 1.0 with leaky features")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def evaluate_threshold(y_true, y_proba, threshold):
    """
    Evaluate model performance at a specific threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    threshold : float
        Classification threshold
    
    Returns:
    --------
    dict : Dictionary containing metrics at this threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    })
    
    return metrics


def plot_threshold_analysis(threshold_results):
    """
    Create comprehensive visualization of threshold analysis.
    
    Parameters:
    -----------
    threshold_results : list of dict
        List of metric dictionaries from evaluate_threshold
    """
    df_thresholds = pd.DataFrame(threshold_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1-Score vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(df_thresholds['threshold'], df_thresholds['precision'], 
             marker='o', label='Precision', linewidth=2)
    ax1.plot(df_thresholds['threshold'], df_thresholds['recall'], 
             marker='s', label='Recall', linewidth=2)
    ax1.plot(df_thresholds['threshold'], df_thresholds['f1_score'], 
             marker='^', label='F1-Score', linewidth=2)
    ax1.set_xlabel('Threshold', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Performance Metrics vs Threshold', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([df_thresholds['threshold'].min() - 0.05, 
                   df_thresholds['threshold'].max() + 0.05])
    
    # Plot 2: Accuracy and Specificity vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(df_thresholds['threshold'], df_thresholds['accuracy'], 
             marker='o', label='Accuracy', linewidth=2, color='green')
    ax2.plot(df_thresholds['threshold'], df_thresholds['specificity'], 
             marker='d', label='Specificity', linewidth=2, color='orange')
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Accuracy & Specificity vs Threshold', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([df_thresholds['threshold'].min() - 0.05, 
                   df_thresholds['threshold'].max() + 0.05])
    
    # Plot 3: Confusion Matrix Components
    ax3 = axes[1, 0]
    x = np.arange(len(df_thresholds))
    width = 0.2
    ax3.bar(x - 1.5*width, df_thresholds['true_positives'], width, 
            label='True Positives', color='#2ecc71')
    ax3.bar(x - 0.5*width, df_thresholds['true_negatives'], width, 
            label='True Negatives', color='#3498db')
    ax3.bar(x + 0.5*width, df_thresholds['false_positives'], width, 
            label='False Positives', color='#e74c3c')
    ax3.bar(x + 1.5*width, df_thresholds['false_negatives'], width, 
            label='False Negatives', color='#f39c12')
    ax3.set_xlabel('Threshold', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Confusion Matrix Components', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_thresholds['threshold'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for _, row in df_thresholds.iterrows():
        table_data.append([
            f"{row['threshold']:.2f}",
            f"{row['accuracy']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1_score']:.3f}"
        ])
    
    table = ax4.table(cellText=table_data,
                      colLabels=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best F1 score row
    best_f1_idx = df_thresholds['f1_score'].idxmax() + 1
    for i in range(5):
        table[(best_f1_idx, i)].set_facecolor('#f1c40f')
        table[(best_f1_idx, i)].set_alpha(0.3)
    
    ax4.set_title('Threshold Comparison Table\n(Highlighted: Best F1-Score)', 
                  fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


print(f"\n{'='*60}")
print("Threshold Optimization Analysis")
print(f"{'='*60}")

# Test multiple thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
y_proba = results["y_proba"]

# Collect metrics for all thresholds
threshold_results = []
for threshold in thresholds:
    metrics = evaluate_threshold(y_test, y_proba, threshold)
    threshold_results.append(metrics)

# Create summary DataFrame
df_thresholds = pd.DataFrame(threshold_results)

# Display summary table
print("\nThreshold Performance Summary:")
print("=" * 100)
print(df_thresholds.to_string(index=False))

# Find best threshold by different criteria
print(f"\n{'='*60}")
print("Best Thresholds by Different Criteria:")
print(f"{'='*60}")
print(f"Best F1-Score:    {df_thresholds.loc[df_thresholds['f1_score'].idxmax(), 'threshold']:.2f} "
      f"(F1 = {df_thresholds['f1_score'].max():.3f})")
print(f"Best Accuracy:    {df_thresholds.loc[df_thresholds['accuracy'].idxmax(), 'threshold']:.2f} "
      f"(Acc = {df_thresholds['accuracy'].max():.3f})")
print(f"Best Precision:   {df_thresholds.loc[df_thresholds['precision'].idxmax(), 'threshold']:.2f} "
      f"(Prec = {df_thresholds['precision'].max():.3f})")
print(f"Best Recall:      {df_thresholds.loc[df_thresholds['recall'].idxmax(), 'threshold']:.2f} "
      f"(Rec = {df_thresholds['recall'].max():.3f})")

# Plot comprehensive analysis
plot_threshold_analysis(threshold_results)

# Recommendation based on business context
print(f"\n{'='*60}")
print("Threshold Selection Recommendation:")
print(f"{'='*60}")
print("""
For LOAN DEFAULT prediction, consider:
- High RECALL (catch defaults): Lower threshold (0.2-0.3) → Minimize False Negatives
- High PRECISION (avoid false alarms): Higher threshold (0.5-0.6) → Minimize False Positives
- BALANCED approach: Use F1-optimal threshold → Balance both metrics

Business Context:
- False Negative (missed default) = Lost money, higher risk
- False Positive (rejected good borrower) = Lost opportunity, customer dissatisfaction

Recommended: Start with F1-optimal threshold, then adjust based on business cost analysis.
""")

# ============================================================================
# ALTERNATIVE MODELS FOR PERFORMANCE IMPROVEMENT
# ============================================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

print(f"\n{'='*60}")
print("Training Alternative Models")
print(f"{'='*60}")

# Dictionary to store all model results
model_results = {}
model_results['Logistic Regression'] = results

# Random Forest Model
print("\n1. Random Forest Classifier")
print("-" * 60)
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor_clean),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

rf_results = train_and_evaluate_model(
    rf_model, 
    X_train_clean,
    X_test_clean,
    y_train,
    y_test,
    model_name="Random Forest"
)
model_results['Random Forest'] = rf_results

# Gradient Boosting Model
print("\n2. Gradient Boosting Classifier")
print("-" * 60)
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor_clean),
    ('classifier', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

gb_results = train_and_evaluate_model(
    gb_model, 
    X_train_clean, 
    X_test_clean, 
    y_train, 
    y_test,
    model_name="Gradient Boosting"
)
model_results['Gradient Boosting'] = gb_results

# Decision Tree Model (for comparison)
print("\n3. Decision Tree Classifier")
print("-" * 60)
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor_clean),
    ('classifier', DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ))
])

dt_results = train_and_evaluate_model(
    dt_model, 
    X_train_clean, 
    X_test_clean, 
    y_train, 
    y_test,
    model_name="Decision Tree"
)
model_results['Decision Tree'] = dt_results

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print(f"\n{'='*60}")
print("Model Performance Comparison")
print(f"{'='*60}")

# Create comparison DataFrame
comparison_data = []
for model_name, result in model_results.items():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = result['y_pred']
    comparison_data.append({
        'Model': model_name,
        'ROC-AUC': result['roc_auc'],
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values('ROC-AUC', ascending=False)

print("\n" + df_comparison.to_string(index=False))

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar plot of ROC-AUC scores
ax1 = axes[0]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax1.bar(df_comparison['Model'], df_comparison['ROC-AUC'], color=colors[:len(df_comparison)])
ax1.set_ylabel('ROC-AUC Score', fontsize=11)
ax1.set_title('Model Comparison: ROC-AUC Scores', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold')

# Radar chart for multiple metrics
ax2 = axes[1]
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(categories)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]

ax2 = plt.subplot(122, projection='polar')

for idx, row in df_comparison.iterrows():
    values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    ax2.fill(angles, values, alpha=0.15)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 1)
ax2.set_title('Multi-Metric Model Comparison', fontsize=12, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print("Best Model:")
print(f"{'='*60}")
best_model_name = df_comparison.iloc[0]['Model']
best_roc_auc = df_comparison.iloc[0]['ROC-AUC']
print(f"{best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

print(f"\n{'='*60}")
print("Analysis Complete!")
print(f"{'='*60}")