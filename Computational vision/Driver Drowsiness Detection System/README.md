# Driver Drowsiness Detection System

This exercise was conducted on its own to explore the possibilities of improving driving safety using computer vision techniques. 

## Problem Statement and Approaches

### First Approach: Training a Personalized Model
Initially, we considered that it would be ideal to train a custom model using a specific dataset to detect drowsiness. This would include features such as:

- Eyes closed
- Yawning
- Head tilts

For this approach, we would have needed:

1. **Labeled Data Set**: Images or videos labeled with the drowsiness conditions.
2. **Model Training**: A training process that includes data preprocessing, feature selection, model training, and performance evaluation.

### Final Approach: Pre-Trained Model and Threshold-Based Detection

We decided to go with a much more lightweight and well-performing system, using a pre-trained model (the shape predictor of dlib) to obtain facial landmarks. We then computed derived features and applied predefined thresholds to determine the drowsiness state.

## How does it work?
The system detects two main conditions:

1. **Eyes Closed for more than 3 seconds**: We calculate the eye aspect ratio (EAR) using the facial landmarks. If the eyes remain closed for more than 3 seconds, an alarm is triggered.
2. **Yawns**: We use the mouth aspect ratio (MAR) to detect yawns. If a yawn is detected, an alarm is triggered.

## Possible Improvements

1. **Y yawning refinement**: Add other filters to detect yawns, not simply opening the mouth.
2. **Implement Nod Detection**: Detect head movements associated with drowsiness.
3. **More Elaborate Alarms**: Include different types of alarms and notifications depending on the type of drowsiness detected.