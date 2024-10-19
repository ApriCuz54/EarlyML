import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pretrained model
MODEL = load_model("bestmodel.h5")

LABEL = {
    0: "Zero", 1: "One", 2: "Two",
    3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight",
    9: "Nine"
}

def preprocess_image(image):
    img_pil = Image.open(image)
    img_pil = img_pil.convert("L")  # Convert to grayscale
    img_pil = img_pil.resize((28, 28))
    img_arr = np.array(img_pil)
    img_arr = img_arr.astype(np.float32) / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)
    return img_arr

# Path to the image file
image_file = "/Users/apricuz/Downloads/six.jpg"

# Preprocess the image
processed_image = preprocess_image(image_file)

# Predict using the model
pred = MODEL.predict(processed_image)[0]
label = LABEL[np.argmax(pred)]

print("Predictions:", pred)
print("Predicted Label:", label)
