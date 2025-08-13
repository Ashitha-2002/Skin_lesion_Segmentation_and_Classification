import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from django.conf import settings
import os
import math

class LesionClassifier:
    def __init__(self):
        self.classification_model_path = 'models/50_efficientnet_model_bal.keras'
        self.segmentation_model_path = 'models/50_epochs_BCDUnet_model.keras'

        self.classification_model = None
        self.segmentation_model = None

        self.class_names = [
            'Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis',
            'Dermatofibroma', 'Melanoma', 'Melanocytic nevus',
            'Squamous cell carcinoma', 'Vascular lesion'
        ]

        self.model_input_size = None
        self.load_models()

    def load_models(self):
        try:
            self.classification_model = tf.keras.models.load_model(
                self.classification_model_path, compile=False
            )
            self.segmentation_model = tf.keras.models.load_model(
                self.segmentation_model_path, compile=False
            )
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading classification models: {e}")

    def bl_resize(self, original_img, new_h, new_w):
        old_h, old_w, c = original_img.shape
        resized = np.zeros((new_h, new_w, c))
        w_scale_factor = old_w / new_w if new_w != 0 else 0
        h_scale_factor = old_h / new_h if new_h != 0 else 0
        for i in range(new_h):
            for j in range(new_w):
                x = i * h_scale_factor
                y = j * w_scale_factor
                x_floor = math.floor(x)
                x_ceil = min(old_h - 1, math.ceil(x))
                y_floor = math.floor(y)
                y_ceil = min(old_w - 1, math.ceil(y))
                if (x_ceil == x_floor) and (y_ceil == y_floor):
                    q = original_img[int(x), int(y), :]
                elif (x_ceil == x_floor):
                    q1 = original_img[int(x), int(y_floor), :]
                    q2 = original_img[int(x), int(y_ceil), :]
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                elif (y_ceil == y_floor):
                    q1 = original_img[int(x_floor), int(y), :]
                    q2 = original_img[int(x_ceil), int(y), :]
                    q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
                else:
                    v1 = original_img[x_floor, y_floor, :]
                    v2 = original_img[x_ceil, y_floor, :]
                    v3 = original_img[x_floor, y_ceil, :]
                    v4 = original_img[x_ceil, y_ceil, :]
                    q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                    q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                resized[i, j, :] = q
        return resized.astype(np.uint8)

    def apply_clahe(self, red_img_arr):
        image_lab = cv2.cvtColor(red_img_arr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        colorimage_l = clahe.apply(image_lab[:, :, 0])
        colorimage_clahe = np.stack(
            (colorimage_l, image_lab[:, :, 1], image_lab[:, :, 2]), axis=2
        )
        image_rgb = cv2.cvtColor(colorimage_clahe, cv2.COLOR_LAB2BGR)
        return image_rgb

    def Hair_removal(self, image_clahe):
        grayScale = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(1, (17, 17))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(image_clahe, thresh2, 1, cv2.INPAINT_TELEA)
        return Image.fromarray(dst)

    def preprocess_image(self, image_path, target_size=(256, 256)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.bl_resize(image, target_size[0], target_size[1])
        image_clahe = self.apply_clahe(image)
        image = self.Hair_removal(image_clahe)
        return np.array(image)

    def classify_lesion(self, image_path):
        try:
            processed_image = self.preprocess_image(image_path).astype(np.float32)
            processed_image = np.expand_dims(processed_image, axis=0)
            predictions = self.classification_model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            return predicted_class, confidence
        except Exception as e:
            print(f"Error in classification: {e}")
            return 'Error', 0.0

    def generate_segmentation_mask(self, image_path):
        try:
            processed_image = self.preprocess_image(image_path)
            img_for_model = np.expand_dims(processed_image, axis=0)
            predictions = self.segmentation_model.predict(img_for_model, batch_size=1, verbose=0)
            predictions = np.where(predictions > 0.5, 1, 0).astype(np.uint8) * 255
            mask = np.squeeze(predictions[0])
            ROI = processed_image.copy()
            img2 = np.stack((mask, mask, mask), axis=2)
            ROI[img2 == 0] = 255
            segmented_region = ROI
            mask_pil = Image.fromarray(mask)
            segmented_pil = Image.fromarray(segmented_region)
            return mask_pil, segmented_pil
        except Exception as e:
            print(f"Error in segmentation: {e}")
            return None, None
