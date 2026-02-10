import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Optional

class EnhancedGeneratorConfig:
    """Enhanced configuration for generator model"""
    def __init__(self):
        self.ws = 256  # Window size
        self.filters = 64  # Number of filters
        self.filter_size = 3  # Filter size
        self.batch_size = 32
        self.scale = 2  # Scale factor for 3-band input
        self.layers = 5  # Number of layers
        self.learning_rate = 0.0001
        self.gpu_growth = True  # GPU memory growth

class EnhancedSolarPanelGenerator:
    """Enhanced TensorFlow 2.x compatible generator with all layers"""
    
    def __init__(self, config: EnhancedGeneratorConfig):
        self.config = config
        
        # Setup GPU
        self._setup_gpu()
        
        # Initialize variable lists
        self.alpha = []
        self.beta = []
        self.bi = []
        self.bo = []
        self.Wo = []
        self.Wi = []
        self.Wi3 = []
        self.Wi8 = []
        self.Wii = []
        self.bii = []
        
        # Build the model
        self.model = self._build_enhanced_model()
        
    def _setup_gpu(self):
        """Setup GPU configuration"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and self.config.gpu_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU setup complete. Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"⚠️ GPU setup warning: {e}")
    
    def _build_enhanced_model(self) -> tf.keras.Model:
        """Build the enhanced generator model with all layers"""
        
        # Input layers
        x8 = tf.keras.layers.Input(shape=(self.config.ws, self.config.ws, 8), name='x8_input')
        x3 = tf.keras.layers.Input(shape=(self.config.scale * self.config.ws, 
                                       self.config.scale * self.config.ws, 3), name='x3_input')
        label_distance = tf.keras.layers.Input(shape=(self.config.ws, self.config.ws, 1), name='label_distance')
        
        # Initialize variables for all layers
        for i in range(self.config.layers):
            # Alpha and beta for convex combination
            self.alpha.append(tf.Variable(0.9, name=f'alpha_{i}'))
            self.beta.append(tf.keras.layers.Lambda(
                lambda x, alpha=alpha: tf.clip_by_value(alpha, 0.0, 1.0), 
                name=f'beta_{i}'
            ))
            
            # Bias terms
            self.bi.append(tf.Variable(tf.zeros([self.config.filters]), name=f'bi_{i}'))
            self.bo.append(tf.Variable(tf.zeros([self.config.filters]), name=f'bo_{i}'))
            self.bii.append(tf.Variable(tf.zeros([self.config.filters]), name=f'bii_{i}'))
            
            # Weight matrices
            self.Wo.append(tf.Variable(
                tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                    shape=[self.config.filter_size, self.config.filter_size, 1, self.config.filters]
                ), name=f'Wo_{i}'
            ))
            
            self.Wi3.append(tf.Variable(
                tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                    shape=[self.config.filter_size, self.config.filter_size, 3, self.config.filters]
                ), name=f'Wi_{i}l3'
            ))
            
            self.Wi8.append(tf.Variable(
                tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                    shape=[self.config.filter_size, self.config.filter_size, 8, self.config.filters]
                ), name=f'Wi_{i}l8'
            ))
            
            self.Wii.append(tf.Variable(
                tf.keras.initializers.TruncatedNormal(stddev=0.1)(
                    shape=[self.config.filter_size, self.config.filter_size, self.config.filters, self.config.filters]
                ), name=f'Wii_{i}'
            ))
        
        # Build layers
        z_layers = []
        zz_layers = []
        outlayer = []
        labelout = []
        
        for i in range(self.config.layers):
            # Convolution operations for 3-band and 8-band inputs
            z3 = tf.keras.layers.Conv2D(
                filters=self.config.filters,
                kernel_size=self.config.filter_size,
                strides=self.config.scale,
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                name=f'conv3_{i}'
            )(x3)
            
            z8 = tf.keras.layers.Conv2D(
                filters=self.config.filters,
                kernel_size=self.config.filter_size,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                name=f'conv8_{i}'
            )(x8)
            
            if i == 0:
                # First layer: combine z3 and z8
                z = tf.keras.layers.Add(name=f'add_{i}')([z3, z8])
                z = tf.keras.layers.Add(name=f'bias_add1_{i}')([z, self.bi[i]])
                z = tf.keras.layers.ReLU(name=f'relu1_{i}')(z)
                z = tf.keras.layers.Add(name=f'bias_add2_{i}')([z, self.bo[i]])
                
            else:
                # Subsequent layers: include previous layer output
                z_prev = outlayer[i-1]
                z_i = tf.keras.layers.Conv2D(
                    filters=self.config.filters,
                    kernel_size=self.config.filter_size,
                    strides=1,
                    padding='same',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                    name=f'conv_i_{i}'
                )(z_prev)
                
                z = tf.keras.layers.Add(name=f'add_{i}')([z3, z8, z_i])
                z = tf.keras.layers.Add(name=f'bias_add1_{i}')([z, self.bi[i]])
                z = tf.keras.layers.ReLU(name=f'relu1_{i}')(z)
                z = tf.keras.layers.Add(name=f'bias_add2_{i}')([z, self.bo[i]])
            
            z_layers.append(z)
            
            # Additional convolution layer with Wii and bii
            zz = tf.keras.layers.Conv2D(
                filters=self.config.filters,
                kernel_size=self.config.filter_size,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                name=f'conv_zz_{i}'
            )(z)
            zz = tf.keras.layers.Add(name=f'bias_add_zz_{i}')([zz, self.bii[i]])
            zz = tf.keras.layers.ReLU(name=f'relu_zz_{i}')(zz)
            
            zz_layers.append(zz)
            
            # Output layer (transpose convolution)
            label_out = tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=self.config.filter_size,
                strides=1,
                padding='same',
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                name=f'deconv_{i}'
            )(zz)
            
            labelout.append(label_out)
            
            if i == 0:
                out = label_out
            else:
                # Convex combination measures impact of layer
                beta_val = self.beta[i](self.alpha[i])
                term1 = tf.keras.layers.Multiply(name=f'mul1_{i}')([beta_val, label_out])
                term2 = tf.keras.layers.Multiply(name=f'mul2_{i}')([1.0 - beta_val, outlayer[i-1]])
                out = tf.keras.layers.Add(name=f'combine_{i}')([term1, term2])
                out = tf.keras.layers.ReLU(name=f'out_relu_{i}')(out)
            
            outlayer.append(out)
        
        # Create model
        model = tf.keras.Model(
            inputs=[x8, x3, label_distance],
            outputs=outlayer[-1],
            name='enhanced_solar_panel_generator'
        )
        
        return model
    
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
        )
        
        return self.model
    
    def compute_layer_losses(self, label_distance, outlayer):
        """Compute losses for each layer"""
        layer_losses = []
        for i, out in enumerate(outlayer):
            loss = tf.reduce_sum(tf.pow(out - label_distance, 2))
            layer_losses.append(loss)
        return layer_losses
    
    def train_step(self, x8_data, x3_data, label_data):
        """Enhanced training step with layer-wise losses"""
        with tf.GradientTape() as tape:
            predictions = self.model([x8_data, x3_data, label_data], training=True)
            
            # Compute main loss
            main_loss = tf.keras.losses.mse(label_data, predictions)
            
            # Compute layer-wise losses for additional regularization
            layer_outputs = []
            for i in range(self.config.layers):
                # Get intermediate layer outputs (simplified for this example)
                layer_outputs.append(predictions)
            
            layer_losses = self.compute_layer_losses(label_data, layer_outputs)
            total_loss = main_loss + 0.1 * tf.reduce_sum(layer_losses)
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'layer_losses': layer_losses
        }
    
    def predict(self, x8_data, x3_data):
        """Generate predictions"""
        dummy_label = tf.zeros((tf.shape(x8_data)[0], self.config.ws, self.config.ws, 1))
        return self.model([x8_data, x3_data, dummy_label], training=False)
    
    def get_model_summary(self):
        """Print detailed model summary"""
        print("=" * 80)
        print("ENHANCED SOLAR PANEL GENERATOR MODEL SUMMARY")
        print("=" * 80)
        self.model.summary()
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Configuration: {self.config.layers} layers, {self.config.filters} filters")
        print("=" * 80)

def create_enhanced_generator(config: EnhancedGeneratorConfig = None) -> EnhancedSolarPanelGenerator:
    """Factory function to create enhanced generator"""
    if config is None:
        config = EnhancedGeneratorConfig()
    
    generator = EnhancedSolarPanelGenerator(config)
    generator.compile_model()
    
    return generator

# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = EnhancedGeneratorConfig()
    config.ws = 256
    config.filters = 64
    config.layers = 3  # Reduced for testing
    config.batch_size = 4
    
    print("🚀 Creating Enhanced Solar Panel Generator...")
    
    # Create and compile model
    generator = create_enhanced_generator(config)
    
    # Print model summary
    generator.get_model_summary()
    
    # Create example data
    batch_size = 2
    x8_example = np.random.rand(batch_size, config.ws, config.ws, 8).astype(np.float32)
    x3_example = np.random.rand(batch_size, config.scale * config.ws, 
                             config.scale * config.ws, 3).astype(np.float32)
    label_example = np.random.rand(batch_size, config.ws, config.ws, 1).astype(np.float32)
    
    print(f"\n📊 Testing with data shapes:")
    print(f"  x8: {x8_example.shape}")
    print(f"  x3: {x3_example.shape}")
    print(f"  label: {label_example.shape}")
    
    # Test forward pass
    try:
        output = generator.model.predict([x8_example, x3_example, label_example], verbose=0)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
        # Test training step
        loss_dict = generator.train_step(x8_example, x3_example, label_example)
        print(f"✅ Training step successful!")
        print(f"  Total loss: {loss_dict['total_loss'].numpy():.6f}")
        print(f"  Main loss: {loss_dict['main_loss'].numpy():.6f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
