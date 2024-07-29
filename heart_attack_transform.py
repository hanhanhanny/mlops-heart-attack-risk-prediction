
import tensorflow as tf
import tensorflow_transform as tft
import os

# Set a custom temporary directory
os.environ['TF_TFT_TMP_DIR'] = '/path/to/your/temp/dir'

def transformed_key(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    

    Description:
        - apply one hot encoding to categorical features
        - apply standardization to float features and int features that are not binary
        - apply renaming of transformed features except for one hot encoded features
    """
    
    outputs = {}

    # Standardize numerical features
    outputs[transformed_key("Age")] = tf.cast(inputs["Age"], tf.int64)
    outputs[transformed_key("Cholesterol")] = tft.scale_to_0_1(inputs["Cholesterol"])
    outputs[transformed_key("Triglycerides")] = tft.scale_to_0_1(inputs["Triglycerides"])
    outputs[transformed_key("Income")] = tft.scale_to_0_1(inputs["Income"])
    outputs[transformed_key("Heart_Rate")] = tft.scale_to_0_1(inputs["Heart Rate"])
    outputs[transformed_key("Stress_Level")] = tft.scale_to_0_1(inputs["Stress Level"])
    outputs[transformed_key("Physical_Activity_Days_Per_Week")] = tft.scale_to_0_1(inputs["Physical Activity Days Per Week"])
    outputs[transformed_key("Sleep_Hours_Per_Day")] = tft.scale_to_0_1(inputs["Sleep Hours Per Day"])

    # Binary features (no transformation)
    outputs["Smoking"] = inputs["Smoking"]
    outputs["Diabetes"] = inputs["Diabetes"]
    outputs["Family_History"] = inputs["Family History"]
    outputs["Obesity"] = inputs["Obesity"]
    outputs["Alcohol_Consumption"] = inputs["Alcohol Consumption"]
    outputs["Previous_Heart_Problems"] = inputs["Previous Heart Problems"]
    outputs["Medication_Use"] = inputs["Medication Use"]

    # Target feature
    outputs["Heart_Attack_Risk"] = tf.cast(inputs["Heart Attack Risk"], tf.int64)

    return outputs
