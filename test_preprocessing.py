import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

def test_preprocessing_pipeline(image_path):
    """Test the preprocessing pipeline step by step"""

    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return

    # Load original image
    image = cv2.imread(image_path)
    original_array = np.array(image)
    print(f"🖼️ Original image shape: {original_array.shape}")

    # Import classifier
    try:
        from lesion_analyzer.ml_utils import LesionClassifier
    except ImportError as e:
        print(f"❌ Failed to import LesionClassifier: {e}")
        return

    classifier = LesionClassifier()

    # Explicitly load models
    try:
        classifier.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        import traceback
        traceback.print_exc()  # show full error


    # Preprocessing test
    try:
        processed_image = classifier.preprocess_image(image_path)
        if processed_image is not None:
            print(f"✅ Processed image shape: {processed_image.shape}")
        else:
            print("❌ Preprocessing failed: returned None")
            return
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return

    # Classification test using classify_lesion()
    try:
        predicted_class, confidence = classifier.classify_lesion(image_path)
        print(f"✅ Classification result: {predicted_class} ({confidence:.2%} confidence)")
    except Exception as e:
        print(f"❌ Classification failed: {e}")

    # Segmentation model test
    if classifier.segmentation_model is not None:
        try:
            mask_image = classifier.generate_segmentation_mask(image_path)
            if mask_image is not None:
                print(f"✅ Segmentation mask generated: {mask_image.size}")
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(original_array, cv2.COLOR_BGR2RGB))
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(mask_image, cmap='gray')
                plt.title("Predicted Mask")
                plt.axis("off")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
    else:
        print("❌ Segmentation model not loaded")


def create_test_image():
    """Create a simple test image if none exists"""
    test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_img)
    test_path = "test_skin_lesion.jpg"
    test_image.save(test_path)
    print(f"🧪 Created dummy test image: {test_path}")
    return test_path

if __name__ == "__main__":
    # Ask for the uploaded image path
    uploaded_image_path = input("Enter path to uploaded image: ").strip()

    if not os.path.exists(uploaded_image_path):
        print(f"❌ Uploaded image not found: {uploaded_image_path}")
    else:
        print(f"✅ Using uploaded image: {uploaded_image_path}")
        test_preprocessing_pipeline(uploaded_image_path)

