import tensorflow as tf
import numpy as np
import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from enhanced_generator import EnhancedSolarPanelGenerator, EnhancedGeneratorConfig

class ModernTrainingConfig:
    """Modern training configuration"""
    def __init__(self):
        self.ws = 256
        self.filters = 64
        self.layers = 5
        self.batch_size = 32
        self.scale = 2
        self.max_layer_steps = 1000
        self.learning_rate = 0.0001
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'
        self.save_interval = 100
        self.validation_split = 0.2

class DataGenerator:
    """Modern data generator for training"""
    
    def __init__(self, config: ModernTrainingConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.ws = config.ws
        self.scale = config.scale
        
    def get_random_input(self, labels: np.ndarray, im_raw8: np.ndarray, 
                      im_raw3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random input batch"""
        batch_size = self.batch_size
        
        # Random batch indices
        indices = np.random.randint(0, len(labels), batch_size)
        
        # Initialize batch arrays
        x8_batch = np.zeros((batch_size, self.ws, self.ws, 8), dtype=np.float32)
        x3_batch = np.zeros((batch_size, self.scale * self.ws, self.scale * self.ws, 3), dtype=np.float32)
        label_batch = np.zeros((batch_size, self.ws, self.ws, 1), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            # Extract random patches
            h8, w8 = im_raw8[idx].shape[:2]
            h3, w3 = im_raw3[idx].shape[:2]
            
            # Random crop positions
            y8 = np.random.randint(0, max(1, h8 - self.ws))
            x8 = np.random.randint(0, max(1, w8 - self.ws))
            
            y3 = np.random.randint(0, max(1, h3 - self.scale * self.ws))
            x3 = np.random.randint(0, max(1, w3 - self.scale * self.ws))
            
            # Extract patches
            x8_batch[i] = im_raw8[idx][y8:y8+self.ws, x8:x8+self.ws, :]
            x3_batch[i] = im_raw3[idx][y3:y3+self.scale*self.ws, x3:x3+self.scale*self.ws, :]
            label_batch[i] = labels[idx][y8:y8+self.ws, x8:x8+self.ws, :]
        
        return x8_batch, x3_batch, label_batch
    
    def create_dataset(self, labels: np.ndarray, im_raw8: np.ndarray, 
                    im_raw3: np.ndarray, training: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset"""
        
        def data_generator():
            while True:
                x8_batch, x3_batch, label_batch = self.get_random_input(labels, im_raw8, im_raw3)
                yield x8_batch, x3_batch, label_batch
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.ws, self.ws, 8), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.scale * self.ws, self.scale * self.ws, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.ws, self.ws, 1), dtype=tf.float32)
            )
        )
        
        if training:
            dataset = dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)
        
        return dataset.batch(self.batch_size)

class ModernTrainer:
    """Modern TensorFlow 2.x trainer"""
    
    def __init__(self, config: ModernTrainingConfig):
        self.config = config
        self.generator_config = EnhancedGeneratorConfig()
        
        # Copy training config to generator config
        for attr in ['ws', 'filters', 'layers', 'batch_size', 'scale', 'learning_rate']:
            setattr(self.generator_config, attr, getattr(config, attr))
        
        # Create generator
        self.generator = EnhancedSolarPanelGenerator(self.generator_config)
        self.generator.compile_model()
        
        # Setup directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Training state
        self.global_step = 0
        self.training_history = []
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, 'model_{epoch:02d}.h5'),
                save_best_only=True,
                save_weights_only=True,
                monitor='loss',
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.log_dir,
                histogram_freq=1,
                write_graph=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=50,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7
            )
        ]
    
    def layer_wise_training(self, labels: np.ndarray, im_raw8: np.ndarray, 
                         im_raw3: np.ndarray) -> Dict[str, List[float]]:
        """Layer-wise training similar to original approach"""
        
        data_gen = DataGenerator(self.config)
        training_history = {'layer_losses': [], 'total_losses': []}
        
        print(f"🚀 Starting layer-wise training for {self.config.layers} layers...")
        
        for layer_idx in range(self.config.layers):
            print(f"\n📊 Training Layer {layer_idx + 1}/{self.config.layers}")
            
            layer_losses = []
            
            # Phase 1: Train individual layer (max_layer_steps)
            print(f"  Phase 1: Training individual layer ({self.config.max_layer_steps} steps)")
            for step in range(self.config.max_layer_steps):
                x8_batch, x3_batch, label_batch = data_gen.get_random_input(labels, im_raw8, im_raw3)
                
                # Train step for specific layer
                loss_dict = self.generator.train_step(x8_batch, x3_batch, label_batch)
                layer_losses.append(loss_dict['main_loss'].numpy())
                
                self.global_step += 1
                
                if step % 100 == 0:
                    print(f"    Step {step}: Loss = {loss_dict['main_loss'].numpy():.6f}")
            
            # Phase 2: Full model training (5 * max_layer_steps)
            print(f"  Phase 2: Full model training ({5 * self.config.max_layer_steps} steps)")
            for step in range(5 * self.config.max_layer_steps):
                x8_batch, x3_batch, label_batch = data_gen.get_random_input(labels, im_raw8, im_raw3)
                
                loss_dict = self.generator.train_step(x8_batch, x3_batch, label_batch)
                layer_losses.append(loss_dict['main_loss'].numpy())
                
                self.global_step += 1
                
                if step % 500 == 0:
                    print(f"    Step {step}: Loss = {loss_dict['main_loss'].numpy():.6f}")
            
            # Save checkpoint for this layer
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f'boundary_{layer_idx}')
            self.generator.model.save_weights(checkpoint_path)
            print(f"  ✅ Saved checkpoint: {checkpoint_path}")
            
            training_history['layer_losses'].extend(layer_losses)
        
        # Final checkpoint
        final_checkpoint = os.path.join(self.config.checkpoint_dir, 'boundary')
        self.generator.model.save_weights(final_checkpoint)
        print(f"\n✅ Final checkpoint saved: {final_checkpoint}")
        
        return training_history
    
    def modern_training(self, labels: np.ndarray, im_raw8: np.ndarray, 
                      im_raw3: np.ndarray, epochs: int = 100) -> Dict[str, List[float]]:
        """Modern Keras-style training"""
        
        data_gen = DataGenerator(self.config)
        
        # Create training dataset
        train_dataset = data_gen.create_dataset(labels, im_raw8, im_raw3, training=True)
        
        # Create validation dataset (split from training data)
        val_size = int(len(labels) * self.config.validation_split)
        train_size = len(labels) - val_size
        
        print(f"🚀 Starting modern training...")
        print(f"  Training samples: {train_size}")
        print(f"  Validation samples: {val_size}")
        print(f"  Epochs: {epochs}")
        
        # Train model
        history = self.generator.model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history.history
    
    def evaluate_model(self, labels: np.ndarray, im_raw8: np.ndarray, 
                    im_raw3: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        
        data_gen = DataGenerator(self.config)
        test_dataset = data_gen.create_dataset(labels, im_raw8, im_raw3, training=False)
        
        # Evaluate on test data
        results = self.generator.model.evaluate(test_dataset, verbose=0)
        
        return dict(zip(self.generator.model.metrics_names, results))
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history"""
        
        plt.figure(figsize=(12, 8))
        
        if 'loss' in history:
            plt.subplot(2, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        if 'mae' in history:
            plt.subplot(2, 2, 2)
            plt.plot(history['mae'], label='Training MAE')
            plt.title('Training MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        
        if 'val_loss' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        if 'val_mae' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['val_mae'], label='Validation MAE')
            plt.title('Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved: {save_path}")
        
        plt.show()

def create_sample_data(num_samples: int = 100, ws: int = 256, scale: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sample data for testing"""
    
    labels = np.random.rand(num_samples, ws, ws, 1).astype(np.float32)
    im_raw8 = np.random.rand(num_samples, ws, ws, 8).astype(np.float32)
    im_raw3 = np.random.rand(num_samples, scale * ws, scale * ws, 3).astype(np.float32)
    
    return labels, im_raw8, im_raw3

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = ModernTrainingConfig()
    config.ws = 128  # Smaller for testing
    config.layers = 3
    config.max_layer_steps = 50  # Reduced for testing
    config.batch_size = 4
    
    print("🚀 Creating Modern Training System...")
    
    # Create sample data
    print("📊 Creating sample data...")
    labels, im_raw8, im_raw3 = create_sample_data(num_samples=20, ws=config.ws, scale=config.scale)
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  8-band images shape: {im_raw8.shape}")
    print(f"  3-band images shape: {im_raw3.shape}")
    
    # Create trainer
    trainer = ModernTrainer(config)
    
    # Choose training method
    print("\n🎯 Choose training method:")
    print("1. Layer-wise training (like original)")
    print("2. Modern Keras training")
    
    # For demo, use layer-wise training
    print("\n🚀 Starting Layer-wise Training...")
    history = trainer.layer_wise_training(labels, im_raw8, im_raw3)
    
    # Plot results
    print("\n📊 Plotting training history...")
    trainer.plot_training_history(history, save_path='./training_history.png')
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    results = trainer.evaluate_model(labels, im_raw8, im_raw3)
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Checkpoints saved in: {config.checkpoint_dir}")
    print(f"📊 Logs saved in: {config.log_dir}")
