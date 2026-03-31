import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from calories import calorie_dict

# Load trained model
model = load_model("food_model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Convert index → class name
class_names = list(class_indices.keys())

# Load test image (PUT YOUR IMAGE NAME HERE)
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

food = class_names[class_index]
confidence = np.max(prediction)

# Get calories
calories = calorie_dict.get(food, "Unknown")

# Output
print("🍽 Food:", food)
print("🔥 Calories:", calories, "kcal")
print("📊 Confidence:", round(confidence*100, 2), "%")