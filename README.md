# Traffic Accident Severity Prediction

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Development](#model-development)
5. [Streamlit Deployment](#streamlit-deployment)
6. [Technologies Used](#technologies-used)
7. [Setup and Installation](#setup-and-installation)
8. [Usage Instructions](#usage-instructions)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview
This project predicts the severity of traffic accidents using machine learning models based on geospatial, weather, and road-related features. The application is deployed using Streamlit, providing an interactive and user-friendly interface for predictions and geospatial visualizations.

## Features
- Predicts accident severity levels:
  1. Minor Impact
  2. Moderate Impact
  3. Severe Impact
  4. Critical Impact
- Geospatial map visualization of accident locations.
- Dynamic input handling with feature alignment.
- Supports Random Forest and XGBoost models for robust predictions.

## Dataset
The dataset used for this project contains detailed records of traffic accidents, including:
- Geospatial data (latitude, longitude)
- Weather conditions
- Environmental factors (temperature, visibility, etc.)
- Accident severity labels

### Source
The dataset can be accessed [here](https://drive.google.com/file/d/1edKrdWNOcgbAo2JtckX-PEyM0FdEq4EG/view?usp=drive_link).

## Model Development
### Data Preprocessing
1. **Cleaning**:
   - Handled missing values and removed duplicates.
2. **Feature Engineering**:
   - One-hot encoded weather conditions.
   - Normalized continuous features.

### Models Used
1. **Random Forest**:
   - Tuned using GridSearchCV.
2. **XGBoost**:
   - Optimized for performance and memory usage.

### Training
- Train-test split: 75%-15%-10%.
- Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).

## Streamlit Deployment
The application is deployed using [Streamlit Cloud], allowing users to:
1. Input accident features dynamically.
2. View predictions from trained models.
3. Visualize accident locations on an interactive map.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `folium`, `joblib`
- **Deployment**: Streamlit Cloud

## Setup and Installation
### Prerequisites
- Python 3.12 or higher
- Virtual environment (optional but recommended)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/traffic-accident-severity.git
   cd traffic-accident-severity
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage Instructions
1. Open the deployed application in your browser or local Streamlit instance.
2. Enter values for the features such as latitude, longitude, temperature, and weather conditions.
3. View the predicted severity and visualize the accident location on a map.
4. Optionally, download the map as an HTML file.

## Results
### Model Performance
| Model            | Accuracy | F1-Score |
|------------------|----------|----------|
| XGBoost          | 89.2%    | 0.90     |

### Example Prediction
| Feature            | Value         |
|--------------------|---------------|
| Start Latitude     | 34.0522       |
| Start Longitude    | -118.2437     |
| Weather Condition  | Clear         |
| Temperature (F)    | 70.0          |
| Predicted Severity | Moderate Impact |

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.


