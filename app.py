from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json
import logging
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)
# Enable CORS with proper configuration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoodPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.mood_categories = ['relaxed', 'happy', 'fine', 'anxious', 'sad', 'angry']
        # Set a random seed based on current timestamp
        np.random.seed(int(pd.Timestamp.now().timestamp()))
        
    def train(self, X, y):
        """
        Train the model with the prepared data
        
        Parameters:
        X (pd.DataFrame): Feature matrix with one-hot encoded days
        y (pd.Series): Target variable with encoded moods
        """
        try:
            logger.info("Training model...")
            # Make sure y is 1-dimensional
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            elif isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.flatten()
                
            self.model.fit(X, y)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def prepare_data(self, mood_logs):
        try:
            logger.info(f"Preparing data from {len(mood_logs)} mood logs")
            
            # Log a sample of the input data
            logger.info(f"Sample mood log: {mood_logs[0] if mood_logs else 'No logs'}")
            
            np.random.seed(int(pd.Timestamp.now().timestamp()))
            
            # Create DataFrame and ensure required columns exist
            df = pd.DataFrame(mood_logs)
            if 'timestamp' not in df.columns:
                raise ValueError("Input data must contain 'timestamp' field")
            if 'mood' not in df.columns:
                raise ValueError("Input data must contain 'mood' field")
            if 'activities' not in df.columns:
                df['activities'] = [[] for _ in range(len(df))]
                
            # Convert timestamp strings to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df = df.sort_values('timestamp', ascending=False)
            
            logger.info(f"Created DataFrame with columns: {df.columns.tolist()}")

            # Determine reference dates
            try:
                most_recent = df['timestamp'].max()
                current_date = pd.Timestamp.now()
                if pd.notna(most_recent) and most_recent.tzinfo:
                    current_date = current_date.tz_localize(most_recent.tzinfo)
                    
                current_date = current_date.date()
            except Exception as e:
                logger.error(f"Error processing dates: {str(e)}")
                # Use current date as fallback
                current_date = pd.Timestamp.now().date()
                
            # Get Monday of the current week
            current_week_monday = current_date - pd.Timedelta(days=current_date.weekday())
            current_week_start = pd.Timestamp.combine(current_week_monday, datetime.min.time())
            
            # Add timezone if timestamps in data have timezone
            if 'timestamp' in df.columns and not df.empty and pd.notna(df['timestamp'].iloc[0]) and df['timestamp'].iloc[0].tzinfo:
                current_week_start = current_week_start.tz_localize(df['timestamp'].iloc[0].tzinfo)
            
            # Exclude data from current week
            df = df[df['timestamp'] < current_week_start]
            
            # Include data from four weeks before the current week
            four_weeks_ago = current_week_start - pd.Timedelta(days=28)
            
            logger.info(f"Date range for analysis: {four_weeks_ago} to {current_week_start}")
            logger.info(f"Data points after date filtering: {len(df)}")

            processed_data = []

            for day in self.days_of_week:
                day_data = df[(df['day_of_week'] == day) & (df['timestamp'] >= four_weeks_ago)]
                
                if not day_data.empty:
                    logger.info(f"Found {len(day_data)} entries for {day}")
                    unique_moods = day_data['mood'].unique()
                    selected_mood = None
                    selected_activities = []

                    if len(unique_moods) >= 3:
                        # Use the latest mood
                        selected_mood = day_data.iloc[0]['mood']
                        # Get activities from the latest mood entry
                        activities = day_data.iloc[0].get('activities', [])
                        if isinstance(activities, list) and activities:
                            # Randomly select two activities from the list
                            if len(activities) >= 2:
                                selected_activities = list(np.random.choice(activities, size=2, replace=False))
                            else:
                                selected_activities = activities
                    else:
                        # Get the most common mood
                        mood_counts = day_data['mood'].value_counts()
                        if not mood_counts.empty:
                            selected_mood = mood_counts.index[0]
                            
                            # Get all instances of the most common mood
                            mood_specific_data = day_data[day_data['mood'] == selected_mood]
                            
                            if not mood_specific_data.empty:
                                # Collect ALL activities for the most common mood
                                all_activities = []
                                for row in mood_specific_data.itertuples():
                                    activities = getattr(row, 'activities', [])
                                    if isinstance(activities, list) and activities:
                                        all_activities.extend(activities)
                                
                                if all_activities:
                                    # Count activity frequencies
                                    activity_counts = pd.Series(all_activities).value_counts()
                                    
                                    # Get the maximum frequency
                                    max_freq = activity_counts.iloc[0]
                                    
                                    if max_freq > 1:
                                        # Get all activities that have the maximum frequency
                                        most_common = activity_counts[activity_counts == max_freq]
                                        if len(most_common) >= 2:
                                            # If there are 2 or more activities with the same max frequency
                                            selected_activities = list(most_common.index[:2])
                                        else:
                                            # If only one most common activity
                                            selected_activities = [most_common.index[0]]
                                    else:
                                        # If no clear common activity, randomly select two activities
                                        unique_activities = list(set(all_activities))
                                        if len(unique_activities) >= 2:
                                            selected_activities = list(np.random.choice(unique_activities, size=2, replace=False))
                                        else:
                                            selected_activities = unique_activities

                    processed_data.append({
                        'day_of_week': day,
                        'mood': selected_mood.lower() if selected_mood else 'unknown',
                        'major_activities': selected_activities
                    })
                else:
                    processed_data.append({
                        'day_of_week': day,
                        'mood': 'unknown',
                        'major_activities': []
                    })

            processed_df = pd.DataFrame(processed_data)
            logger.info(f"Processed data: {processed_df.to_dict('records')}")

            # Make sure all mood categories are included in label encoder
            self.label_encoder.fit(self.mood_categories + ['unknown'])
            
            # Transform the mood column to numerical values
            processed_df['mood_encoded'] = self.label_encoder.transform(processed_df['mood'])
            
            # Create one-hot encoded features for days of week
            X = pd.get_dummies(processed_df['day_of_week'])
            X = X.reindex(columns=self.days_of_week, fill_value=0)
            
            # Ensure y is 1-dimensional
            y = processed_df['mood_encoded'].values

            self.daily_activities = dict(zip(processed_df['day_of_week'], processed_df['major_activities']))

            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def predict_weekly_moods(self):
        try:
            test_data = pd.DataFrame(index=self.days_of_week)
            test_X = pd.get_dummies(test_data.index)
            test_X = test_X.reindex(columns=self.days_of_week, fill_value=0)

            logger.info(f"Prediction input shape: {test_X.shape}")
            predictions = self.model.predict(test_X)
            logger.info(f"Raw predictions: {predictions}")
            
            predicted_moods = self.label_encoder.inverse_transform(predictions)
            logger.info(f"Predicted moods: {predicted_moods}")

            weekly_predictions = {}
            for day, mood in zip(self.days_of_week, predicted_moods):
                activities = self.daily_activities.get(day, [])
                weekly_predictions[day] = {
                    'mood': mood if mood != 'unknown' else 'No prediction available',
                    'activities': activities
                }

            return {'daily_predictions': weekly_predictions}

        except Exception as e:
            logger.error(f"Error in predict_weekly_moods: {str(e)}")
            raise

def predict_mood(mood_logs):
    try:
        # Validate input data
        if len(mood_logs) < 7:
            return {
                'error': 'Need at least one week of mood data for predictions'
            }
            
        predictor = MoodPredictor()
        X, y = predictor.prepare_data(mood_logs)
        predictor.train(X, y)
        predictions = predictor.predict_weekly_moods()
        return predictions
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {'error': str(e)}

# Mock database for demonstration
_mock_user_data = {}

def get_user_mood_logs(user_id):
    """
    In a real implementation, this would query your database
    This is a mock function to simulate database access
    """
    # Return mock data if we have it
    if user_id in _mock_user_data:
        return _mock_user_data[user_id]
        
    # Create some sample data for testing
    moods = ["happy", "relaxed", "fine", "anxious", "sad"]
    activity_options = [
        ["exercise", "reading"], 
        ["work", "socializing"], 
        ["family", "tv"], 
        ["meditation", "cooking"]
    ]
    
    sample_data = []
    for i in range(30):
        date = datetime.now() - timedelta(days=i)
        sample_data.append({
            "mood": np.random.choice(moods),
            "timestamp": date.isoformat(),
            "activities": np.random.choice(activity_options).tolist()
        })
    
    _mock_user_data[user_id] = sample_data
    return sample_data

@app.route('/api/predict-mood', methods=['GET'])
def predict_route():
    try:
        # In a real app, you'd get the user ID from authentication
        # For this example, we'll use a query parameter
        user_id = request.args.get('user_id', 'default_user')
        
        # Get the user's mood logs (30 days by default)
        days = request.args.get('days', 30, type=int)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # In a real implementation, fetch from database
        mood_logs = get_user_mood_logs(user_id)
        
        logger.info(f"Retrieved {len(mood_logs)} mood logs for user {user_id}")
        logger.info(f"Sample log: {mood_logs[0] if mood_logs else 'No logs'}")
        
        # Check if we have sufficient data
        if len(mood_logs) < 7:
            return jsonify({
                'success': True,
                'predictions': {},
                'message': 'Need at least one week of mood data for predictions'
            })
            
        result = predict_mood(mood_logs)
        
        if 'error' in result:
            return jsonify({
                'success': False, 
                'message': result['error']
            }), 500
            
        return jsonify({
            'success': True,
            'predictions': result.get('daily_predictions', {}),
            'insights': result.get('insights', {})
        })
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Server error while generating predictions: {str(e)}'
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == "__main__":
    logger.info("Starting Mood Prediction API server")
    app.run(host='0.0.0.0', port=5000, debug=True)