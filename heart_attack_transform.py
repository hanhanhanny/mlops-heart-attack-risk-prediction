
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

    # Convert "Female" and "Male" to 1 and 0
    gender_lookup = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(['Female', 'Male']),
            values=tf.constant([1, 0]),
        ),
        default_value=tf.constant(-1),
    )
    outputs[transformed_key("Sex")] = tf.cast(gender_lookup.lookup(inputs["Sex"]), tf.int64) 

    # Standardize numerical features
    outputs[transformed_key("Cholesterol")] = tft.scale_to_0_1(inputs["Cholesterol"])
    outputs[transformed_key("Sedentary_Hours_Per_Day")] = tft.scale_to_0_1(inputs["Sedentary Hours Per Day"])
    outputs[transformed_key("BMI")] = tft.scale_to_0_1(inputs["BMI"])
    outputs[transformed_key("Exercise_Hours_Per_Week")] = tft.scale_to_0_1(inputs["Exercise Hours Per Week"])


    # Binary features (no transformation)
    outputs["Age"] = inputs["Age"]
    outputs["Heart_Rate"] = inputs["Heart Rate"]
    outputs["Triglycerides"] = inputs["Triglycerides"]
    outputs["Stress_Level"] = inputs["Stress Level"]
    outputs["Physical_Activity_Days_Per_Week"] = inputs["Physical Activity Days Per Week"]
    outputs["Sleep_Hours_Per_Day"] = inputs["Sleep Hours Per Day"]
    outputs["Smoking"] = inputs["Smoking"]
    outputs["Diabetes"] = inputs["Diabetes"]
    outputs["Family_History"] = inputs["Family History"]
    outputs["Obesity"] = inputs["Obesity"]
    outputs["Income"] = inputs["Income"]
    outputs["Alcohol_Consumption"] = inputs["Alcohol Consumption"]
    outputs["Previous_Heart_Problems"] = inputs["Previous Heart Problems"]
    outputs["Medication_Use"] = inputs["Medication Use"]

    # Target feature
    outputs[transformed_key("Heart Attack Risk")] = tf.cast(inputs["Heart Attack Risk"], tf.int64)

    return outputs
