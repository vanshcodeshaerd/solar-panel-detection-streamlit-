import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple

class GeneratorConfig:
    """Configuration class for generator model"""
    def __init__(self):
        self.ws = 256  # Window size
        self.filters = 64  # Number of filters
        self.filter_size = 3  # Filter size
        self.batch_size = 32
        self.scale = 2  # Scale factor for 3-band input
        self.layers = 5  # Number of layers
        self.learning_rate = 0.0001

class SolarPanelGenerator:
    """TensorFlow 2.x compatible generator for solar panel detection"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.alpha = []
        self.beta = []
        self.bi = []
        self.bo = []
        self.Wo = []
        self.Wi = []
        self.Wi3 = None
        self.Wi8 = None
        
        # Build the model
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the generator model using Keras API"""
        
        # Input layers
        x8 = tf.keras.layers.Input(shape=(self.config.ws, self.config.ws, 8), name='x8_input')
        x3 = tf.keras.layers.Input(shape=(self.config.scale * self.config.ws, 
                                       self.config.scale * self.config.ws, 3), name='x3_input')
        label_distance = tf.keras.layers.Input(shape=(self.config.ws, self.config.ws, 1), name='label_distance')
        
        # Initialize variables
        for i in range(self.config.layers):
            self.alpha.append(tf.Variable(0.9, name=f'alpha_{i}'))
            self.beta.append(tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, 0.0, 1.0), 
                name=f'beta_{i}'
            ))
            self.bi.append(tf.Variable(tf.zeros([self.config.filters]), name=f'bi_{i}'))
            self.bo.append(tf.Variable(tf.zeros([self.config.filters]), name=f'bo_{i}'))
            self.Wo.append(tf.Variable(
                tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                    shape=[self.config.filter_size, self.config.filter_size, 1, self.config.filters]
                ), name=f'Wo_{i}'
            ))
        
        # First layer special handling
        self.Wi3 = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                shape=[self.config.filter_size, self.config.filter_size, 3, self.config.filters]
            ), name='Wi_0l3'
        )
        self.Wi8 = tf.Variable(
            tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                shape=[self.config.filter_size, self.config.filter_size, 8, self.config.filters]
            ), name='Wi_0l8'
        )
        
        # Build layers
        z_layers = []
        outlayer = []
        labelout = []
        
        for i in range(self.config.layers):
            if i == 0:
                # First layer: project 11 bands onto one distance transform band
                z3 = tf.keras.layers.Conv2D(
                    filters=self.config.filters,
                    kernel_size=self.config.filter_size,
                    strides=self.config.scale,
                    padding='same',
                    name=f'conv3_{i}'
                )(x3)
                
                z8 = tf.keras.layers.Conv2D(
                    filters=self.config.filters,
                    kernel_size=self.config.filter_size,
                    strides=1,
                    padding='same',
                    name=f'conv8_{i}'
                )(x8)
                
                z = tf.keras.layers.Add(name=f'add_{i}')([z3, z8])
                z = tf.keras.layers.ReLU(name=f'relu1_{i}')(z)
                z = tf.keras.layers.BatchNormalization(name=f'bn1_{i}')(z)
                z = tf.keras.layers.ReLU(name=f'relu2_{i}')(z)
                
            else:
                # Non-initial bands are perturbations of previous bands output
                z = tf.keras.layers.Conv2D(
                    filters=self.config.filters,
                    kernel_size=self.config.filter_size,
                    strides=1,
                    padding='same',
                    name=f'conv_{i}'
                )(outlayer[i-1])
                z = tf.keras.layers.ReLU(name=f'relu1_{i}')(z)
                z = tf.keras.layers.BatchNormalization(name=f'bn1_{i}')(z)
                z = tf.keras.layers.ReLU(name=f'relu2_{i}')(z)
            
            z_layers.append(z)
            
            # Output layer (transpose convolution)
            label_out = tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=self.config.filter_size,
                strides=1,
                padding='same',
                name=f'deconv_{i}'
            )(z)
            
            labelout.append(label_out)
            
            if i == 0:
                out = label_out
            else:
                # Convex combination measures impact of layer
                beta_val = self.beta[i](self.alpha[i])
                out = tf.keras.layers.Add(name=f'combine_{i}')([
                    tf.keras.layers.Multiply()([beta_val, label_out]),
                    tf.keras.layers.Multiply()([1.0 - beta_val, outlayer[i-1]])
                ])
                out = tf.keras.layers.ReLU(name=f'out_relu_{i}')(out)
            
            outlayer.append(out)
        
        # Create model
        model = tf.keras.Model(
            inputs=[x8, x3, label_distance],
            outputs=outlayer[-1],
            name='solar_panel_generator'
        )
        
        return model
    
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae']
        )
        
        return self.model
    
    def train_step(self, x8_data, x3_data, label_data):
        """Single training step"""
        with tf.GradientTape() as tape:
            predictions = self.model([x8_data, x3_data, label_data], training=True)
            loss = tf.keras.losses.mse(label_data, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def predict(self, x8_data, x3_data):
        """Generate predictions"""
        # Create dummy label for prediction (not used in forward pass)
        dummy_label = tf.zeros((tf.shape(x8_data)[0], self.config.ws, self.config.ws, 1))
        return self.model([x8_data, x3_data, dummy_label], training=False)

def create_generator(config: GeneratorConfig = None) -> SolarPanelGenerator:
    """Factory function to create generator"""
    if config is None:
        config = GeneratorConfig()
    
    generator = SolarPanelGenerator(config)
    generator.compile_model()
    
    return generator

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = GeneratorConfig()
    config.ws = 256
    config.filters = 64
    config.layers = 5
    
    # Create and compile model
    generator = create_generator(config)
    
    # Print model summary
    print("Model Summary:")
    generator.model.summary()
    
    # Example training data shapes
    batch_size = 4
    x8_example = np.random.rand(batch_size, config.ws, config.ws, 8).astype(np.float32)
    x3_example = np.random.rand(batch_size, config.scale * config.ws, 
                             config.scale * config.ws, 3).astype(np.float32)
    label_example = np.random.rand(batch_size, config.ws, config.ws, 1).astype(np.float32)
    
    # Test forward pass
    try:
        output = generator.model.predict([x8_example, x3_example, label_example])
        print(f"✅ Model test successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
