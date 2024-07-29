
import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os
from keras_tuner.engine import base_tuner
from keras_tuner import RandomSearch, HyperParameters
from tfx.components.trainer.fn_args_utils import FnArgs
from typing import NamedTuple, Dict, Text, Any

LABEL_KEY = "Heart Attack Risk"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def get_hyperparameters() -> HyperParameters:
    """Returns hyperparameters for building model"""
    hp = HyperParameters()
    hp.Int('units', min_value=32, max_value=512, step=32, default=128)
    hp.Int('num_layers', min_value=1, max_value=4, step=1, default=3)
    hp.Float('learning_rate', min_value=1e-4, max_value=1e-1, sampling='LOG', default=1e-3)
    return hp

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=64)->tf.data.Dataset:
    """Get post_transform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)
    dataset = dataset.shuffle(buffer_size=10000)
    return dataset

def model_builder(hparams: HyperParameters):
    """Build machine learning model"""
    inputs = {
        transformed_name('Age'): tf.keras.Input(shape=(1,), name=transformed_name('Age'), dtype=tf.int64),
        transformed_name('Cholesterol'): tf.keras.Input(shape=(1,), name=transformed_name('Cholesterol'), dtype=tf.int64),
        transformed_name('Triglycerides'): tf.keras.Input(shape=(1,), name=transformed_name('Triglycerides'), dtype=tf.int64),
        transformed_name('Income'): tf.keras.Input(shape=(1,), name=transformed_name('Income'), dtype=tf.int64),
        transformed_name('Heart_Rate'): tf.keras.Input(shape=(1,), name=transformed_name('Heart_Rate'), dtype=tf.int64),
        transformed_name('Stress_Level'): tf.keras.Input(shape=(1,), name=transformed_name('Stress_Level'), dtype=tf.int64),
        transformed_name('Physical_Activity_Days_Per_Week'): tf.keras.Input(shape=(1,), name=transformed_name('Physical_Activity_Days_Per_Week'), dtype=tf.int64),
        transformed_name('Sleep_Hours_Per_Day'): tf.keras.Input(shape=(1,), name=transformed_name('Sleep_Hours_Per_Day'), dtype=tf.int64),
        'Smoking': tf.keras.Input(shape=(1,), name='Smoking', dtype=tf.int64),
        'Diabetes': tf.keras.Input(shape=(1,), name='Diabetes', dtype=tf.int64),
        'Family_History': tf.keras.Input(shape=(1,), name='Family_History', dtype=tf.int64),
        'Obesity': tf.keras.Input(shape=(1,), name='Obesity', dtype=tf.int64),
        'Alcohol_Consumption': tf.keras.Input(shape=(1,), name='Alcohol_Consumption', dtype=tf.int64),
        'Previous_Heart_Problems': tf.keras.Input(shape=(1,), name='Previous_Heart_Problems', dtype=tf.int64),
        'Medication_Use': tf.keras.Input(shape=(1,), name='Medication_Use', dtype=tf.int64)
    }
    
    # Combine all inputs into a single tensor
    concatenated_inputs = layers.Concatenate()(list(inputs.values()))

    x = layers.Dense(hparams.get('units'), activation='relu')(concatenated_inputs)
    for _ in range(hparams.get('num_layers') - 1):
        x = layers.Dense(hparams.get('units') // 2, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.get('learning_rate')),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    model.summary()
    return model

# Tuner component will run this function
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', RandomSearch),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
        A namedtuple contains the following:
        - tuner: A RandomSearch tuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model, e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
    """
    hp = get_hyperparameters()
    # Define tuner
    tuner = RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=30,
        directory=fn_args.working_dir,
        project_name='heart_attack_risk_classification',
        hyperparameters=hp
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, 10)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': eval_set
        }
    )

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
        
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
    
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    hp = get_hyperparameters()
    model = model_builder(hp)
    
    # Train the model
    model.fit(
        x=train_set,
        validation_data=eval_set,
        callbacks=[tensorboard_callback, es, mc],
        steps_per_epoch=100, 
        validation_steps=100,
        epochs=10
    )
    
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }
    
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
