"""
Human-in-the-Loop Feedback System

Provides mechanisms for collecting and integrating human feedback into the verification process.
"""

from .collector import FeedbackCollector, FeedbackType, FeedbackItem
from .analyzer import FeedbackAnalyzer, FeedbackPattern
from .integrator import FeedbackIntegrator, LearningUpdate
from .interface import FeedbackInterface, WebInterface, CLIInterface

__all__ = [
    'FeedbackCollector',
    'FeedbackType',
    'FeedbackItem',
    'FeedbackAnalyzer',
    'FeedbackPattern',
    'FeedbackIntegrator',
    'LearningUpdate',
    'FeedbackInterface',
    'WebInterface',
    'CLIInterface'
]