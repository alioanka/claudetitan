"""
Machine Learning Module for Strategy Improvement
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import settings

logger = logging.getLogger(__name__)

@dataclass
class MLModel:
    """ML model structure"""
    name: str
    model: Any
    scaler: StandardScaler
    feature_columns: List[str]
    target_column: str
    accuracy: float
    created_at: datetime
    last_trained: datetime
    performance_metrics: Dict[str, float]

class FeatureEngineer:
    """Feature engineering for trading data"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = StandardScaler()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical and statistical features"""
        try:
            df = data.copy()
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # EMA features
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Price relative to moving averages
            df['price_sma5_ratio'] = df['close'] / df['sma_5']
            df['price_sma20_ratio'] = df['close'] / df['sma_20']
            df['price_sma50_ratio'] = df['close'] / df['sma_50']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_ratio'] = df['atr'] / df['close']
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price_trend'] = df['volume'] * df['price_change']
            
            # Volatility
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_10'] = df['price_change'].rolling(10).std()
            df['volatility_20'] = df['price_change'].rolling(20).std()
            
            # Momentum indicators
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Support and Resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            
            # Trend features
            df['trend_5'] = np.where(df['close'] > df['sma_5'], 1, -1)
            df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, -1)
            df['trend_50'] = np.where(df['close'] > df['sma_50'], 1, -1)
            
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                df[f'price_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'price_std_{window}'] = df['close'].rolling(window).std()
                df[f'price_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Define feature columns
            self.feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Created {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return data
    
    def create_target_variable(self, data: pd.DataFrame, future_periods: int = 5) -> pd.DataFrame:
        """Create target variable for prediction"""
        try:
            df = data.copy()
            
            # Calculate future returns
            df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
            
            # Create classification target (1: buy, 0: hold, -1: sell)
            df['target'] = 0  # hold
            df.loc[df['future_return'] > 0.02, 'target'] = 1  # buy
            df.loc[df['future_return'] < -0.02, 'target'] = -1  # sell
            
            # Remove rows with NaN target
            df = df.dropna(subset=['target'])
            
            logger.info(f"Created target variable with distribution: {df['target'].value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variable: {e}")
            return data

class MLModelTrainer:
    """Train and manage ML models"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_training_data(self, market_data: List[Dict], trade_outcomes: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from market data and trade outcomes"""
        try:
            # Convert market data to DataFrame
            df = pd.DataFrame(market_data)
            if df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create features
            df = self.feature_engineer.create_features(df)
            
            # Create target variable
            df = self.feature_engineer.create_target_variable(df)
            
            if df.empty:
                return pd.DataFrame(), pd.Series()
            
            # Prepare features and target
            X = df[self.feature_engineer.feature_columns]
            y = df['target']
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, MLModel]:
        """Train multiple ML models"""
        try:
            if X.empty or y.empty:
                logger.warning("No training data available")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define models
            models_config = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42,
                    class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric='mlogloss'
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'svm': SVC(
                    kernel='rbf',
                    random_state=42,
                    class_weight='balanced',
                    probability=True
                )
            }
            
            trained_models = {}
            
            for name, model in models_config.items():
                try:
                    logger.info(f"Training {name}...")
                    
                    # Train model
                    if name in ['logistic_regression', 'svm']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation score
                    if name in ['logistic_regression', 'svm']:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Create ML model object
                    ml_model = MLModel(
                        name=name,
                        model=model,
                        scaler=scaler if name in ['logistic_regression', 'svm'] else None,
                        feature_columns=self.feature_engineer.feature_columns,
                        target_column='target',
                        accuracy=accuracy,
                        created_at=datetime.now(),
                        last_trained=datetime.now(),
                        performance_metrics={
                            'accuracy': accuracy,
                            'cv_mean': cv_mean,
                            'cv_std': cv_std,
                            'test_samples': len(X_test)
                        }
                    )
                    
                    trained_models[name] = ml_model
                    
                    # Save model
                    self._save_model(ml_model)
                    
                    logger.info(f"{name} trained - Accuracy: {accuracy:.3f}, CV: {cv_mean:.3f} ± {cv_std:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            return trained_models
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def train_lstm_model(self, X: pd.DataFrame, y: pd.Series) -> Optional[MLModel]:
        """Train LSTM model for sequence prediction"""
        try:
            if X.empty or y.empty:
                return None
            
            # Prepare sequence data
            sequence_length = 20
            X_sequences, y_sequences = self._create_sequences(X, y, sequence_length)
            
            if len(X_sequences) == 0:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(3, activation='softmax')  # 3 classes: buy, hold, sell
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            
            # Create ML model object
            ml_model = MLModel(
                name='lstm',
                model=model,
                scaler=scaler,
                feature_columns=self.feature_engineer.feature_columns,
                target_column='target',
                accuracy=test_accuracy,
                created_at=datetime.now(),
                last_trained=datetime.now(),
                performance_metrics={
                    'accuracy': test_accuracy,
                    'loss': test_loss,
                    'epochs': len(history.history['loss']),
                    'test_samples': len(X_test)
                }
            )
            
            # Save model
            self._save_model(ml_model)
            
            logger.info(f"LSTM model trained - Accuracy: {test_accuracy:.3f}")
            return ml_model
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return None
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X.iloc[i-sequence_length:i].values)
                y_sequences.append(y.iloc[i])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def predict_signal(self, model_name: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Predict trading signal using trained model"""
        try:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found")
                return None
            
            model_obj = self.models[model_name]
            
            # Prepare features
            features_df = self.feature_engineer.create_features(market_data)
            if features_df.empty:
                return None
            
            # Get latest features
            latest_features = features_df[self.feature_engineer.feature_columns].iloc[-1:].fillna(0)
            
            # Make prediction
            if model_obj.scaler:
                latest_features_scaled = model_obj.scaler.transform(latest_features)
                prediction = model_obj.model.predict(latest_features_scaled)[0]
                probabilities = model_obj.model.predict_proba(latest_features_scaled)[0]
            else:
                prediction = model_obj.model.predict(latest_features)[0]
                probabilities = model_obj.model.predict_proba(latest_features)[0]
            
            # Convert prediction to signal
            signal_map = {-1: 'sell', 0: 'hold', 1: 'buy'}
            signal = signal_map.get(prediction, 'hold')
            
            # Calculate confidence
            confidence = max(probabilities)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'buy': probabilities[2] if len(probabilities) > 2 else 0,
                    'hold': probabilities[1] if len(probabilities) > 1 else 0,
                    'sell': probabilities[0]
                },
                'model': model_name,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error predicting signal: {e}")
            return None
    
    def _save_model(self, model: MLModel):
        """Save model to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{model.name}.joblib")
            
            # Save model and scaler
            model_data = {
                'model': model.model,
                'scaler': model.scaler,
                'feature_columns': model.feature_columns,
                'target_column': model.target_column,
                'accuracy': model.accuracy,
                'created_at': model.created_at,
                'last_trained': model.last_trained,
                'performance_metrics': model.performance_metrics
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Saved model {model.name} to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model {model.name}: {e}")
    
    def load_models(self) -> Dict[str, MLModel]:
        """Load all saved models"""
        try:
            loaded_models = {}
            
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.joblib'):
                    model_name = filename[:-7]  # Remove .joblib extension
                    model_path = os.path.join(self.model_dir, filename)
                    
                    try:
                        model_data = joblib.load(model_path)
                        
                        ml_model = MLModel(
                            name=model_name,
                            model=model_data['model'],
                            scaler=model_data['scaler'],
                            feature_columns=model_data['feature_columns'],
                            target_column=model_data['target_column'],
                            accuracy=model_data['accuracy'],
                            created_at=model_data['created_at'],
                            last_trained=model_data['last_trained'],
                            performance_metrics=model_data['performance_metrics']
                        )
                        
                        loaded_models[model_name] = ml_model
                        logger.info(f"Loaded model {model_name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading model {model_name}: {e}")
                        continue
            
            self.models = loaded_models
            return loaded_models
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {}
    
    def get_model_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all models"""
        try:
            performance = {}
            
            for name, model in self.models.items():
                performance[name] = {
                    'accuracy': model.accuracy,
                    'created_at': model.created_at.isoformat(),
                    'last_trained': model.last_trained.isoformat(),
                    'metrics': model.performance_metrics
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}

class MLStrategyOptimizer:
    """Optimize trading strategies using ML insights"""
    
    def __init__(self, ml_trainer: MLModelTrainer):
        self.ml_trainer = ml_trainer
        self.optimization_history = []
    
    def optimize_strategy_parameters(self, strategy_name: str, historical_data: pd.DataFrame) -> Dict:
        """Optimize strategy parameters using ML insights"""
        try:
            # This would implement parameter optimization
            # For now, return default parameters
            logger.info(f"Optimizing parameters for {strategy_name}")
            
            return {
                'strategy': strategy_name,
                'optimized_parameters': {},
                'expected_improvement': 0.0,
                'optimization_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing strategy {strategy_name}: {e}")
            return {}
    
    def generate_ensemble_weights(self, model_performances: Dict) -> Dict[str, float]:
        """Generate optimal weights for ensemble models"""
        try:
            # Calculate weights based on model performance
            total_accuracy = sum(perf['accuracy'] for perf in model_performances.values())
            
            weights = {}
            for name, perf in model_performances.items():
                weights[name] = perf['accuracy'] / total_accuracy if total_accuracy > 0 else 1.0 / len(model_performances)
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {name: weight / total_weight for name, weight in weights.items()}
            
            logger.info(f"Generated ensemble weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Error generating ensemble weights: {e}")
            return {}
