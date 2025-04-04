from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import optuna
from xgboost import XGBRegressor

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
PREDICTIONS_FOLDER = 'predictions'
ANALYSIS_FOLDER = 'analysis'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
app.config['ANALYSIS_FOLDER'] = ANALYSIS_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Global variables
ml_model = None
scaler = StandardScaler()
dl_model = None
feature_order = []
stats_html = ""
missing_values_html = ""
histogram_path = None
X_data, y_data = None, None

# Suggested default values and units
feature_suggestions = {
    'temperature': (25, '°C'),
    'pressure': (101.3, 'kPa'),
    'humidity': (50, '%'),
    'wind_speed': (10, 'km/h'),
    'time': (12, 'hr'),
    'concentration': (1.0, 'mol/L'),
    'ph': (7.0, ''),
    'volume': (100, 'mL'),
    'mass': (5.0, 'g'),
    'speed': (60, 'm/s'),
    'density': (1.2, 'g/cm³'),
    'energy': (50, 'J'),
    'length': (10, 'cm'),
    'area': (25, 'cm²'),
    'power': (100, 'W'),
    'voltage': (5, 'V'),
    'current': (2, 'A'),
    'frequency': (60, 'Hz'),
    'resistance': (10, 'Ω')
    # Extend with more features as needed
}

def process_dataset(file_path):
    global feature_order, scaler, stats_html, missing_values_html, histogram_path, X_data, y_data
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".html":
            tables = pd.read_html(file_path, encoding='utf-8')
            if not tables:
                return "Error: No tables found in the uploaded file."
            data = tables[0]
        elif ext == ".csv":
            data = pd.read_csv(file_path)
        elif ext in [".xls", ".xlsx"]:
            data = pd.read_excel(file_path)
        else:
            return "Error: Unsupported file format."
    except Exception as e:
        return f"Error processing file: {e}"

    data = data.iloc[1:].reset_index(drop=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='any')
    data.columns = data.columns.str.lower()

    feature_order = list(data.drop(columns=['score'], errors='ignore').columns)

    if 'score' not in data.columns:
        return "Error: 'score' column not found."

    X = data.drop(columns=['score'])
    y = data['score']
    scaler.fit(X)

    X_data, y_data = X, y

    stats_html = data.describe().to_html(classes='table table-striped')
    missing_values_html = data.isnull().sum().to_frame(name='Missing Values').to_html(classes='table table-striped')

    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=20, kde=True)
    plt.title('Distribution of Score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    histogram_path = os.path.join(STATIC_FOLDER, 'analysis.png')
    plt.savefig(histogram_path)
    plt.close()

    return "Dataset processed successfully."

def optimize_xgboost(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
        }
        model = XGBRegressor(**params, random_state=42)
        score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
        return -np.mean(score)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    best_model = XGBRegressor(**study.best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'model.pkl'))
    return best_model

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return "No file selected."
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        message = process_dataset(path)

        if X_data is not None and y_data is not None:
            global ml_model
            ml_model = optimize_xgboost(X_data, y_data)

        return redirect(url_for('preview'))
    return render_template('upload.html')

@app.route('/preview')
def preview():
    return render_template('preview.html', stats=stats_html, missing_values=missing_values_html, histogram_path=histogram_path)

@app.route('/download_analysis')
def download_analysis():
    return send_file(histogram_path, as_attachment=True)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global ml_model, dl_model

    if ml_model is None:
        ml_path = os.path.join(MODEL_FOLDER, 'model.pkl')
        if os.path.exists(ml_path):
            ml_model = joblib.load(ml_path)

    if dl_model is None:
        dl_path = os.path.join(MODEL_FOLDER, 'model.h5')
        if os.path.exists(dl_path):
            dl_model = load_model(dl_path)

    suggestions = []
    for feature in feature_order:
        if feature in feature_suggestions:
            val, unit = feature_suggestions[feature]
            suggestions.append((feature, val, unit))
        else:
            suggestions.append((feature, '', ''))

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        input_data = []
        row_data = {}
        for feature in feature_order:
            value = request.form.get(feature)
            row_data[feature] = value
            if feature not in ['date', 'id']:
                input_data.append(float(value))

        input_array = np.array([input_data])

        if model_type == 'dl':
            input_scaled = scaler.transform(input_array)
            prediction = dl_model.predict(input_scaled)[0][0]
        else:
            prediction = ml_model.predict(input_array)[0]

        row_data['prediction'] = prediction
        pred_df = pd.DataFrame([row_data])
        pred_path = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')

        if os.path.exists(pred_path):
            existing = pd.read_csv(pred_path)
            pred_df = pd.concat([existing, pred_df], ignore_index=True)

        pred_df.to_csv(pred_path, index=False)

        return render_template('predict.html', prediction=prediction, feature_order=feature_order, model_type=model_type, suggestions=suggestions)

    return render_template('predict.html', prediction=None, feature_order=feature_order, model_type=None, suggestions=suggestions)

@app.route('/download_predictions')
def download_predictions():
    prediction_csv_path = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')
    return send_file(prediction_csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


# -------------------

# from flask import Flask, request, render_template, redirect, url_for, send_file
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# import optuna
# from xgboost import XGBRegressor

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# MODEL_FOLDER = 'models'
# PREDICTIONS_FOLDER = 'predictions'
# ANALYSIS_FOLDER = 'analysis'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MODEL_FOLDER, exist_ok=True)
# os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
# os.makedirs(ANALYSIS_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MODEL_FOLDER'] = MODEL_FOLDER
# app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
# app.config['ANALYSIS_FOLDER'] = ANALYSIS_FOLDER
# app.config['STATIC_FOLDER'] = STATIC_FOLDER

# # Global variables
# ml_model = None
# scaler = StandardScaler()
# dl_model = None
# feature_order = []
# stats_html = ""
# missing_values_html = ""
# histogram_path = None
# X_data, y_data = None, None
# feature_suggestions = []

# def process_dataset(file_path):
#     global feature_order, scaler, stats_html, missing_values_html, histogram_path, X_data, y_data, feature_suggestions
#     ext = os.path.splitext(file_path)[1].lower()

#     try:
#         if ext == ".html":
#             tables = pd.read_html(file_path, encoding='utf-8')
#             if not tables:
#                 return "Error: No tables found in the uploaded file."
#             data = tables[0]
#         elif ext == ".csv":
#             data = pd.read_csv(file_path)
#         elif ext in [".xls", ".xlsx"]:
#             data = pd.read_excel(file_path)
#         else:
#             return "Error: Unsupported file format."
#     except Exception as e:
#         return f"Error processing file: {e}"

#     data = data.iloc[1:].reset_index(drop=True)
#     data = data.apply(pd.to_numeric, errors='coerce')
#     data = data.dropna(axis=1, how='any')
#     data.columns = data.columns.str.lower()

#     feature_order = list(data.drop(columns=['score'], errors='ignore').columns)

#     if 'score' not in data.columns:
#         return "Error: 'score' column not found."

#     X = data.drop(columns=['score'])
#     y = data['score']
#     scaler.fit(X)

#     X_data, y_data = X, y

#     # Generate feature suggestions
#     feature_suggestions = []
#     for col in X.columns:
#         suggestion = X[col].median()
#         unit = "(unit)"  # Replace with actual unit if known
#         feature_suggestions.append((col, round(suggestion, 2), unit))

#     stats_html = data.describe().to_html(classes='table table-striped')
#     missing_values_html = data.isnull().sum().to_frame(name='Missing Values').to_html(classes='table table-striped')

#     plt.figure(figsize=(10, 6))
#     sns.histplot(y, bins=20, kde=True)
#     plt.title('Distribution of Score')
#     plt.xlabel('Score')
#     plt.ylabel('Frequency')
#     histogram_path = os.path.join(STATIC_FOLDER, 'analysis.png')
#     plt.savefig(histogram_path)
#     plt.close()

#     return "Dataset processed successfully."

# def optimize_xgboost(X, y):
#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'gamma': trial.suggest_float('gamma', 0, 5),
#             'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
#             'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
#         }
#         model = XGBRegressor(**params, random_state=42)
#         score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
#         return -np.mean(score)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=30)

#     best_model = XGBRegressor(**study.best_params)
#     best_model.fit(X, y)
#     joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'model.pkl'))
#     return best_model

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file.filename == '':
#             return "No file selected."
#         path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(path)
#         message = process_dataset(path)

#         if X_data is not None and y_data is not None:
#             global ml_model
#             ml_model = optimize_xgboost(X_data, y_data)

#         return redirect(url_for('preview'))
#     return render_template('upload.html')

# @app.route('/preview')
# def preview():
#     return render_template('preview.html', stats=stats_html, missing_values=missing_values_html, histogram_path=histogram_path)

# @app.route('/download_analysis')
# def download_analysis():
#     return send_file(histogram_path, as_attachment=True)

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     global ml_model, dl_model

#     if ml_model is None:
#         ml_path = os.path.join(MODEL_FOLDER, 'model.pkl')
#         if os.path.exists(ml_path):
#             ml_model = joblib.load(ml_path)

#     if dl_model is None:
#         dl_path = os.path.join(MODEL_FOLDER, 'model.h5')
#         if os.path.exists(dl_path):
#             dl_model = load_model(dl_path)

#     if request.method == 'POST':
#         model_type = request.form.get('model_type')
#         input_data = []
#         record = {}

#         for feature in feature_order:
#             value = request.form.get(feature)
#             if feature == 'date':
#                 input_data.append(0)
#                 record[feature] = value
#             elif feature == 'id':
#                 input_data.append(0)
#                 record[feature] = value
#             else:
#                 float_value = float(value)
#                 input_data.append(float_value)
#                 record[feature] = float_value

#         input_array = np.array([input_data])

#         if model_type == 'dl':
#             input_scaled = scaler.transform(input_array)
#             prediction = dl_model.predict(input_scaled)[0][0]
#         else:
#             prediction = ml_model.predict(input_array)[0]

#         record['predicted_score'] = round(prediction, 2)
#         prediction_df = pd.DataFrame([record])
#         prediction_csv_path = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')
#         if os.path.exists(prediction_csv_path):
#             prediction_df.to_csv(prediction_csv_path, mode='a', header=False, index=False)
#         else:
#             prediction_df.to_csv(prediction_csv_path, index=False)

#         return render_template('predict.html', prediction=prediction, feature_order=feature_order, model_type=model_type, suggestions=feature_suggestions)

#     return render_template('predict.html', prediction=None, feature_order=feature_order, model_type=None, suggestions=feature_suggestions)

# @app.route('/download_predictions')
# def download_predictions():
#     prediction_csv_path = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')
#     return send_file(prediction_csv_path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)


# ---------------------------------------


# from flask import Flask, request, render_template, redirect, url_for, send_file
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# import optuna
# from xgboost import XGBRegressor

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# MODEL_FOLDER = 'models'
# PREDICTIONS_FOLDER = 'predictions'
# ANALYSIS_FOLDER = 'analysis'
# STATIC_FOLDER = 'static'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MODEL_FOLDER, exist_ok=True)
# os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
# os.makedirs(ANALYSIS_FOLDER, exist_ok=True)
# os.makedirs(STATIC_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MODEL_FOLDER'] = MODEL_FOLDER
# app.config['PREDICTIONS_FOLDER'] = PREDICTIONS_FOLDER
# app.config['ANALYSIS_FOLDER'] = ANALYSIS_FOLDER
# app.config['STATIC_FOLDER'] = STATIC_FOLDER

# # Global variables
# ml_model = None
# scaler = StandardScaler()
# dl_model = None
# feature_order = []
# feature_suggestions = {}
# feature_unit = {}
# stats_html = ""
# missing_values_html = ""
# histogram_path = None
# X_data, y_data = None, None

# def process_dataset(file_path):
#     global feature_order, scaler, stats_html, missing_values_html, histogram_path, X_data, y_data, feature_suggestions, feature_unit
#     ext = os.path.splitext(file_path)[1].lower()

#     try:
#         if ext == ".html":
#             tables = pd.read_html(file_path, encoding='utf-8')
#             if not tables:
#                 return "Error: No tables found in the uploaded file."
#             data = tables[0]
#         elif ext == ".csv":
#             data = pd.read_csv(file_path)
#         elif ext in [".xls", ".xlsx"]:
#             data = pd.read_excel(file_path)
#         else:
#             return "Error: Unsupported file format."
#     except Exception as e:
#         return f"Error processing file: {e}"

#     data = data.iloc[1:].reset_index(drop=True)
#     data = data.apply(pd.to_numeric, errors='coerce')
#     data = data.dropna(axis=1, how='any')
#     data.columns = data.columns.str.lower()

#     feature_order = list(data.drop(columns=['score'], errors='ignore').columns)

#     # Suggest sample values and dummy units
#     feature_suggestions = {feature: round(data[feature].mean(), 2) for feature in feature_order}
#     feature_unit = {feature: "unit" for feature in feature_order}

#     if 'score' not in data.columns:
#         return "Error: 'score' column not found."

#     X = data.drop(columns=['score'])
#     y = data['score']
#     scaler.fit(X)

#     X_data, y_data = X, y

#     stats_html = data.describe().to_html(classes='table table-striped')
#     missing_values_html = data.isnull().sum().to_frame(name='Missing Values').to_html(classes='table table-striped')

#     plt.figure(figsize=(10, 6))
#     sns.histplot(y, bins=20, kde=True)
#     plt.title('Distribution of Score')
#     plt.xlabel('Score')
#     plt.ylabel('Frequency')
#     histogram_path = os.path.join(STATIC_FOLDER, 'analysis.png')
#     plt.savefig(histogram_path)
#     plt.close()

#     return "Dataset processed successfully."

# def optimize_xgboost(X, y):
#     def objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 300),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#             'gamma': trial.suggest_float('gamma', 0, 5),
#             'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1, log=True),
#             'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1, log=True),
#         }
#         model = XGBRegressor(**params, random_state=42)
#         score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3)
#         return -np.mean(score)

#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=30)

#     best_model = XGBRegressor(**study.best_params)
#     best_model.fit(X, y)
#     joblib.dump(best_model, os.path.join(MODEL_FOLDER, 'model.pkl'))
#     return best_model

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file.filename == '':
#             return "No file selected."
#         path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(path)
#         message = process_dataset(path)

#         if X_data is not None and y_data is not None:
#             global ml_model
#             ml_model = optimize_xgboost(X_data, y_data)

#         return redirect(url_for('preview'))
#     return render_template('upload.html')

# @app.route('/preview')
# def preview():
#     return render_template('preview.html', stats=stats_html, missing_values=missing_values_html, histogram_path=histogram_path)

# @app.route('/download_analysis')
# def download_analysis():
#     return send_file(histogram_path, as_attachment=True)

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     global ml_model, dl_model

#     if ml_model is None:
#         ml_path = os.path.join(MODEL_FOLDER, 'model.pkl')
#         if os.path.exists(ml_path):
#             ml_model = joblib.load(ml_path)

#     if dl_model is None:
#         dl_path = os.path.join(MODEL_FOLDER, 'model.h5')
#         if os.path.exists(dl_path):
#             dl_model = load_model(dl_path)

#     if request.method == 'POST':
#         model_type = request.form.get('model_type')
#         input_data = [float(request.form[feature]) for feature in feature_order]
#         input_array = np.array([input_data])

#         if model_type == 'dl':
#             input_scaled = scaler.transform(input_array)
#             prediction = dl_model.predict(input_scaled)[0][0]
#         else:
#             prediction = ml_model.predict(input_array)[0]

#         return render_template(
#             'predict.html', 
#             prediction=prediction,
#             feature_order=feature_order, 
#             model_type=model_type,
#             suggestions=[(feature, feature_suggestions[feature], feature_unit[feature]) for feature in feature_order]
#         )

#     return render_template(
#         'predict.html', 
#         prediction=None, 
#         feature_order=feature_order, 
#         model_type=None,
#         suggestions=[(feature, feature_suggestions[feature], feature_unit[feature]) for feature in feature_order]
#     )

# @app.route('/download_predictions')
# def download_predictions():
#     prediction_csv_path = os.path.join(PREDICTIONS_FOLDER, 'predictions.csv')
#     return send_file(prediction_csv_path, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True)


