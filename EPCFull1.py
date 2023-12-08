# Import necessary libraries
from flask import Flask, jsonify
from google.cloud import bigquery
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

app = Flask(__name__)

@app.route('/run-model', methods=['POST'])
def run_model():

	# Initialize BigQuery client
	client = bigquery.Client()
	
	# Set BigQuery dataset parameters
	DATASET_ID = 'Vertex'
	TABLE_ID = 'EPCClean'
	
	# Define SQL queries
	sql = f"""
	SELECT
		property_type,
		built_form,
		co2_emissions_current,
		total_floor_area,
		main_fuel_rename,
		uprn,
		construction_year,
		mainheat_env_eff,
		hot_water_env_eff,
		floor_env_eff,
		windows_env_eff,
		walls_env_eff,
		roof_env_eff,
		mainheatc_env_eff,
		lighting_env_eff
	FROM
		`{client.project}.{DATASET_ID}.{TABLE_ID}`
	WHERE NOT (co2_emissions_current <= 0
	  OR (mainheat_env_eff = 'N/A'
		  OR hot_water_env_eff = 'N/A'
		  OR floor_env_eff = 'N/A'
		  OR windows_env_eff = 'N/A'
		  OR walls_env_eff = 'N/A'
		  OR roof_env_eff = 'N/A'
		  OR mainheatc_env_eff = 'N/A'
		  OR lighting_env_eff = 'N/A')
	  OR (mainheat_env_eff IN ('Poor', 'Very Poor')
		  AND property_type <> 'Flat'
		  AND current_energy_rating IN ('A', 'B')
		  AND Lodgement_Date < '2020-01-01'))
	"""

	sql1 = f"""
	SELECT
		property_type,
		built_form,
		co2_emissions_current,
		total_floor_area,
		main_fuel_rename,
		uprn,
		construction_year,
		mainheat_env_eff,
		hot_water_env_eff,
		floor_env_eff,
		windows_env_eff,
		walls_env_eff,
		roof_env_eff,
		mainheatc_env_eff,
		lighting_env_eff
	FROM
		`{client.project}.{DATASET_ID}.{TABLE_ID}`
	WHERE co2_emissions_current <= 0
	  OR (mainheat_env_eff IN ('Poor', 'Very Poor')
		  AND property_type <> 'Flat'
		  AND current_energy_rating IN ('A', 'B')
		  AND Lodgement_Date < '2020-01-01')
	"""
	
	# Load EPC Valid datasets from BigQuery and create table
	epc_valid = client.query(sql).result().to_dataframe()
	epc_valid_table = f"{client.project}.{DATASET_ID}.EPCValid"
	job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_EMPTY, autodetect=True)
	epc_validpayload = client.load_table_from_dataframe(epc_valid, epc_valid_table, job_config=job_config)
	epc_validpayload.result()

	# Load EPC Invalid datasets from BigQuery and create table
	epc_invalid = client.query(sql1).result().to_dataframe()
	epc_invalid_table = f"{client.project}.{DATASET_ID}.EPCInvalid"
	epc_invalidpayload = client.load_table_from_dataframe(epc_invalid, epc_invalid_table, job_config=job_config)
	epc_invalidpayload.result()

	# Split datasets into training and validation datasets
	X = epc_valid.drop(columns=['co2_emissions_current'])
	y = epc_valid['co2_emissions_current']
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)
	
	# Preprocess the data
	categorical_cols = ['property_type', 'built_form', 'main_fuel_rename', 'mainheat_env_eff', 'hot_water_env_eff', 'floor_env_eff', 'windows_env_eff', 'walls_env_eff', 'roof_env_eff', 'mainheatc_env_eff', 'lighting_env_eff']
	numerical_cols = ['total_floor_area', 'construction_year']
	
	categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))
	])
	
	numerical_transformer = Pipeline(steps=[
	('scaler', StandardScaler())
	])
	
	preprocessor = ColumnTransformer(
	transformers=[
	('cat', categorical_transformer, categorical_cols),
	('num', numerical_transformer, numerical_cols)
	]
	)
	
	X_train_transformed = preprocessor.fit_transform(X_train)
	X_validation_transformed = preprocessor.transform(X_validation)
	
	# Hyperparameter tuning
	param_dist = {
	'n_estimators': randint(100, 500),
	'max_depth': [None, 10, 20, 30, 40],
	'min_samples_split': randint(2, 20),
	'min_samples_leaf': randint(1, 10)
	}
	
	# RandomForestRegressor
	rf = RandomForestRegressor()
	
	# RandomizedSearchCV
	random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
	random_search.fit(X_train_transformed, y_train)
	
	best_params = random_search.best_params_
	
	# Train model with the best parameters
	best_rf = RandomForestRegressor(**best_params)
	best_rf.fit(X_train_transformed, y_train)
	
	# Evaluate model
	predictions = best_rf.predict(X_validation_transformed)
	mse = mean_squared_error(y_validation, predictions)
	r2 = r2_score(y_validation, predictions)
	
	# Create dataframe with scores
	scores_df = pd.DataFrame({
	'MSE': [mse],
	'R2_Score': [r2]
	})
	
	# Export MSE and R2 to BigQuery
	validation_table = "Vertex.EPCEvaluation"
	validationpayload = client.load_table_from_dataframe(scores_df, validation_table, job_config=job_config)
	validationpayload.result()
	
	# Predict on EPC Invalid
	epc_invalid_transformed = preprocessor.transform(epc_invalid.drop(columns=['co2_emissions_current']))
	epc_invalid['co2_emissions_predicted'] = best_rf.predict(epc_invalid_transformed)
	
	# Export predictions to BigQuery
	ml_table = "Vertex.EPCInvalidFixed1"
	mlpayload = client.load_table_from_dataframe(epc_invalid, ml_table, job_config=job_config)
	mlpayload.result()
	
	# Return a response
	return jsonify({"message": "ML model execution complete", "MSE": mse, "R2_Score": r2})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
