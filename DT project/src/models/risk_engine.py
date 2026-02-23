# src/models/risk_engine.py
import numpy as np
from datetime import datetime
import json

class RiskScoringEngine:
    def __init__(self):
        # Weights for each factor (sum = 1.0)
        self.weights = {
            'drowsiness': 0.4,
            'distraction': 0.3,
            'emotion': 0.2,
            'seatbelt': 0.1
        }
        
        # Risk thresholds
        self.thresholds = {
            'low': 30,
            'medium': 70,
            'high': 85
        }
        
        # Historical data for trends
        self.risk_history = []
        self.max_history = 100
        
        # Alert cooldown (seconds)
        self.last_alert_time = None
        self.alert_cooldown = 5
        
    def calculate_risk(self, predictions):
        """
        predictions: dict with keys 'drowsiness', 'distraction', 'emotion', 'seatbelt'
        Each value is (class_name, confidence)
        """
        risk_scores = {}
        
        # Drowsiness risk mapping
        drowsiness_map = {
            'awake': 10,
            'drowsy': 60,
            'sleeping': 95
        }
        
        # Distraction risk mapping
        distraction_map = {
            'safe_driving': 5,
            'talking_phone': 50,
            'operating_radio': 40,
            'drinking': 55,
            'reaching_behind': 65,
            'texting': 85
        }
        
        # Emotion risk mapping
        emotion_map = {
            'neutral': 10,
            'happy': 5,
            'surprised': 30,
            'frustrated': 60,
            'angry': 80
        }
        
        # Seatbelt risk
        seatbelt_map = {
            'seatbelt_on': 0,
            'seatbelt_off': 100
        }
        
        # Calculate individual risks
        if 'drowsiness' in predictions:
            cls, conf = predictions['drowsiness']
            risk_scores['drowsiness'] = drowsiness_map.get(cls, 50) * conf
        
        if 'distraction' in predictions:
            cls, conf = predictions['distraction']
            risk_scores['distraction'] = distraction_map.get(cls, 50) * conf
        
        if 'emotion' in predictions:
            cls, conf = predictions['emotion']
            risk_scores['emotion'] = emotion_map.get(cls, 50) * conf
        
        if 'seatbelt' in predictions:
            cls, conf = predictions['seatbelt']
            risk_scores['seatbelt'] = seatbelt_map.get(cls, 50) * conf
        
        # Calculate weighted total
        total_risk = sum(risk_scores.get(k, 0) * self.weights.get(k, 0) 
                        for k in self.weights.keys())
        
        # Apply temporal smoothing
        self.risk_history.append(total_risk)
        if len(self.risk_history) > self.max_history:
            self.risk_history.pop(0)
        
        # Use moving average for stability
        smoothed_risk = np.mean(self.risk_history[-5:])  # Last 5 frames
        
        # Determine risk level
        risk_level = self._get_risk_level(smoothed_risk)
        
        return {
            'total_risk': smoothed_risk,
            'risk_level': risk_level,
            'individual_scores': risk_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_risk_level(self, risk_score):
        if risk_score < self.thresholds['low']:
            return 'LOW'
        elif risk_score < self.thresholds['medium']:
            return 'MEDIUM'
        elif risk_score < self.thresholds['high']:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def should_alert(self, risk_info):
        """Determine if we should trigger an alert"""
        risk_level = risk_info['risk_level']
        current_time = datetime.now()
        
        # Check cooldown
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).seconds
            if time_diff < self.alert_cooldown:
                return False
        
        # Alert based on risk level
        if risk_level in ['HIGH', 'CRITICAL']:
            self.last_alert_time = current_time
            return True
        
        # Alert on rapid risk increase
        if len(self.risk_history) > 10:
            recent_trend = np.mean(self.risk_history[-3:]) - np.mean(self.risk_history[-10:-7])
            if recent_trend > 20:  # 20% increase
                self.last_alert_time = current_time
                return True
        
        return False
    
    def get_alert_level(self, risk_info):
        """Get appropriate alert based on risk level"""
        risk_level = risk_info['risk_level']
        
        alerts = {
            'LOW': {
                'type': 'info',
                'message': 'Safe driving',
                'action': 'none'
            },
            'MEDIUM': {
                'type': 'warning',
                'message': 'Caution: Unsafe behavior detected',
                'action': 'sound_alert'
            },
            'HIGH': {
                'type': 'danger',
                'message': 'DANGER: Immediate attention needed',
                'action': 'sound_alert + vibration'
            },
            'CRITICAL': {
                'type': 'critical',
                'message': 'CRITICAL: Emergency! Taking action',
                'action': 'all_alerts + emergency_contact'
            }
        }
        
        return alerts.get(risk_level, alerts['MEDIUM'])