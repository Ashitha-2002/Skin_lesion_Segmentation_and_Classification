import tensorflow as tf
import numpy as np
import json
import h5py
from pathlib import Path

def diagnose_keras_model(model_path):
    """Diagnose a .keras model file to understand its structure"""
    print(f"Diagnosing model: {model_path}")
    print("=" * 50)
    
    try:
        # Try to load model metadata without full loading
        if model_path.endswith('.keras'):
            # .keras files are ZIP archives, we can peek inside
            import zipfile
            with zipfile.ZipFile(model_path, 'r') as zip_file:
                if 'config.json' in zip_file.namelist():
                    config_data = zip_file.read('config.json')
                    config = json.loads(config_data.decode('utf-8'))
                    
                    print("Model Configuration:")
                    print(f"Name: {config.get('name', 'Unknown')}")
                    print(f"Class: {config.get('class_name', 'Unknown')}")
                    
                    # Look for input layer info
                    layers = config.get('config', {}).get('layers', [])
                    for layer in layers:
                        if layer.get('class_name') == 'InputLayer':
                            batch_shape = layer.get('config', {}).get('batch_shape')
                            if batch_shape:
                                print(f"Expected input shape: {batch_shape}")
                                return batch_shape[1:]  # Remove batch dimension
                    
                    # If no InputLayer found, look in first layer
                    if layers:
                        first_layer = layers[0]
                        print(f"First layer: {first_layer.get('class_name')}")
                        if 'input_shape' in first_layer.get('config', {}):
                            input_shape = first_layer['config']['input_shape']
                            print(f"Input shape from first layer: {input_shape}")
                            return input_shape
        
        # Fallback: try to load and inspect
        print("Trying to load model for inspection...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            print(f"Model loaded successfully!")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            return model.input_shape[1:]  # Remove batch dimension
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
            
    except Exception as e:
        print(f"Error diagnosing model: {e}")
        return None

def test_different_input_shapes(model_path, base_shape=(256, 256, 3)):
    """Test different input shapes to find the correct one"""
    print(f"\nTesting different input shapes for: {model_path}")
    print("=" * 50)
    
    # Common input shapes for medical imaging models
    test_shapes = [
        (224, 224, 3),  # Standard ImageNet
        (257, 257, 3),  # Your current setting
        (299, 299, 3),  # InceptionV3/Xception
        (331, 331, 3),  # NASNet
        (512, 512, 3),  # High resolution
        (128, 128, 3),  # Lower resolution
        (384, 384, 3),  # EfficientNet variants
    ]
    
    for shape in test_shapes:
        try:
            print(f"Testing shape: {shape}")
            
            # Create dummy input
            dummy_input = tf.random.normal((1,) + shape)
            
            # Try to load model and predict
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            output = model(dummy_input)
            
            print(f"✓ SUCCESS with shape {shape}")
            print(f"  Output shape: {output.shape}")
            return shape
            
        except Exception as e:
            error_msg = str(e)
            if "expected axis -1 of input shape to have value" in error_msg:
                # Extract expected value from error
                import re
                match = re.search(r'expected axis -1 of input shape to have value (\d+)', error_msg)
                if match:
                    expected_channels = int(match.group(1))
                    print(f"  ✗ Expected {expected_channels} channels, got {shape[2]}")
            else:
                print(f"  ✗ Failed: {error_msg[:100]}...")
    
    print("No compatible input shape found")
    return None

def create_compatible_model_loader(expected_shape):
    """Create a model loader with the correct input shape"""
    code = f'''
def load_models_with_correct_shape(self):
    """Load models with the correct input shape: {expected_shape}"""
    try:
        classification_path = 'E:/skin_lesion_analyzer/models/50_efficientnet_model1.keras'
        segmentation_path = 'E:/skin_lesion_analyzer/models/30_BCDUnet_model.keras'
        
        if os.path.exists(classification_path):
            self.classification_model = tf.keras.models.load_model(
                classification_path, 
                compile=False, 
                safe_mode=False
            )
            print("Classification model loaded successfully!")
            print(f"Expected input shape: {expected_shape}")
        
        if os.path.exists(segmentation_path):
            self.segmentation_model = tf.keras.models.load_model(
                segmentation_path, 
                compile=False, 
                safe_mode=False
            )
            print("Segmentation model loaded successfully!")
            
    except Exception as e:
        print(f"Error loading models: {{e}}")

def preprocess_image_correct_size(self, image_path, target_size={expected_shape[:2]}):
    """Preprocess image with correct target size"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {{e}}")
        return None
'''
    return code

if __name__ == "__main__":
    model_paths = [
        'E:/skin_lesion_analyzer/models/50_efficientnet_model1.keras',
        'E:/skin_lesion_analyzer/models/30_BCDUnet_model.keras'
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"ANALYZING: {Path(model_path).name}")
            print(f"{'='*60}")
            
            # First, try to diagnose the expected shape
            expected_shape = diagnose_keras_model(model_path)
            
            if expected_shape:
                print(f"Detected expected input shape: {expected_shape}")
            else:
                # If diagnosis fails, try different shapes
                expected_shape = test_different_input_shapes(model_path)
                
            if expected_shape:
                print(f"\n✓ Found working input shape: {expected_shape}")
                print("\nGenerated code for your LesionClassifier:")
                print(create_compatible_model_loader(expected_shape))
            else:
                print(f"\n✗ Could not determine correct input shape for {model_path}")
        else:
            print(f"Model not found: {model_path}")