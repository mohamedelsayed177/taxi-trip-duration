from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from taxi_trip_duration.manipulation.feature_engineering import target_log
from taxi_trip_duration.manipulation.data_helper import (
    load_dataset, 
    drop_id_features, 
    train_val_split, 
    cat_num_features, 
    preprocessing
)

def ridge_model(preprocessor):
    
    model = Ridge(alpha=1.0)
    ridge_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    
    return ridge_pipeline
    
    
if __name__ == "__main__":
    print("#" * 10)
    print("Executing Baseline Ridge Regression Pipeline...")    
    # Load the dataset
    train, val, _ = load_dataset(data_type="split")
    print("#" * 10)
    
    # drop id and vendor_id features
    train = drop_id_features(train)
    val = drop_id_features(val)
    
    # Apply log transformation to the target variable
    train = target_log(train)
    val = target_log(val)
    
    # drop the original datetime column
    train = train.drop(columns=['pickup_datetime'])  # drop the original datetime column
    val = val.drop(columns=['pickup_datetime'])
    
    # Split the train and validation datasets into features and target variable
    X_train, y_train, X_val, y_val = train_val_split(train, val)
    
    # Identify categorical and numerical features
    cat ,num = cat_num_features(X_train)
    
    # Create preprocessing pipelines for both scaling methods and evaluate the Ridge regression model
    for scale in ['standard', 'minmax']:
        print(f"Using {scale} scaling for numerical features.")
        preprocessor = preprocessing(cat,num,scaling=scale)
        
        # Create and fit the Ridge regression model pipeline
        pipeline = ridge_model(preprocessor)
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_val)
        score = r2_score(y_val, preds)
        
        print(f"Baseline Validation R² Score with {scale} scaling: {score:.5f}")
        print("#" * 10)
        
        
'''
##########
Using standard scaling for numerical features.
Baseline Validation R² Score with standard scaling: 0.04400
##########
Using minmax scaling for numerical features.
Baseline Validation R² Score with minmax scaling: 0.03710
##########
'''