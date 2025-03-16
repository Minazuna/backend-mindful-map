import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
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
            self.model.fit(X, y)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def prepare_data(self, mood_logs):
        try:
            np.random.seed(int(pd.Timestamp.now().timestamp()))
            
            df = pd.DataFrame(mood_logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df = df.sort_values('timestamp', ascending=False)

            most_recent = df['timestamp'].max()
            current_date = pd.Timestamp.now(tz=most_recent.tz).date()

            # Always get Monday of the current week
            current_week_monday = current_date - pd.Timedelta(days=current_date.weekday())  

            # Ensure the time is set to the start of Monday
            current_week_start = pd.Timestamp.combine(current_week_monday, datetime.min.time()).tz_localize(most_recent.tz)

            # Exclude data from Monday of this week through Sunday
            df = df[df['timestamp'] < current_week_start]

            # Include data from four weeks before the current week
            four_weeks_ago = current_week_start - pd.Timedelta(days=28)


            processed_data = []

            for day in self.days_of_week:
                seed = int(pd.Timestamp.now().timestamp() * hash(day)) % (2**32 - 1)
                np.random.seed(seed)
                
                day_data = df[(df['day_of_week'] == day) & (df['timestamp'] >= four_weeks_ago)]

                if not day_data.empty:
                    unique_moods = day_data['mood'].unique()
                    selected_mood = None
                    selected_activities = []

                    if len(unique_moods) >= 3:
                        # Use the latest mood
                        selected_mood = day_data.iloc[0]['mood']
                        # Get activities from the latest mood entry
                        activities = day_data.iloc[0]['activities']
                        if isinstance(activities, list) and activities:
                            # Randomly select two activities from the list
                            if len(activities) >= 2:
                                selected_activities = list(np.random.choice(activities, size=2, replace=False))
                            else:
                                selected_activities = activities
                    else:
                        # Get the most common mood
                        mood_counts = day_data['mood'].value_counts()
                        selected_mood = mood_counts.index[0]
                        
                        # Get all instances of the most common mood
                        mood_specific_data = day_data[day_data['mood'] == selected_mood]
                        
                        if not mood_specific_data.empty:
                            # Collect ALL activities for the most common mood
                            all_activities = []
                            for row in mood_specific_data.itertuples():
                                activities = row.activities
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

            self.label_encoder.fit(self.mood_categories + ['unknown'])
            processed_df['mood_encoded'] = self.label_encoder.transform(processed_df['mood'])

            X = pd.get_dummies(processed_df['day_of_week'])
            X = X.reindex(columns=self.days_of_week, fill_value=0)
            y = processed_df['mood_encoded']

            self.daily_activities = dict(zip(processed_df['day_of_week'], processed_df['major_activities']))

            return X, y

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def predict_weekly_moods(self):
        try:
            test_data = pd.DataFrame(index=self.days_of_week)
            test_X = pd.get_dummies(test_data.index)
            test_X = test_X.reindex(columns=self.days_of_week, fill_value=0)

            predictions = self.model.predict(test_X)
            predicted_moods = self.label_encoder.inverse_transform(predictions)

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
        predictor = MoodPredictor()
        X, y = predictor.prepare_data(mood_logs)
        predictor.train(X, y)
        predictions = predictor.predict_weekly_moods()
        return predictions
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    try:
        input_data = sys.stdin.read().strip()
        if not input_data:
            raise ValueError("No input data received")
        
        mood_logs = json.loads(input_data)
        result = predict_mood(mood_logs)
        
        # Ensure proper JSON output
        print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
        sys.stdout.flush()
        
    except Exception as e:
        error_result = {'error': str(e)}
        print(json.dumps(error_result))
        sys.stdout.flush()