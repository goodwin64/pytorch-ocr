import sys
import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
from models.crnn import CRNN  # Assuming CRNN is not dependent on PyTorch
from static_variables import classes
from utils.decoded_answer_list_to_string import decoded_answer_list_to_string

# Load the ONNX model
onnx_path = "crnn.onnx"
onnx_model = onnx.load(onnx_path)

# Create an ONNX Runtime inference session
ort_session = ort.InferenceSession(onnx_path)

# Define input shape
image_width = 250
image_height = 60

# Define transformations
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_width, image_height), resample=Image.BILINEAR)
    image = np.array(image)
    # Convert to grayscale
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Normalize to [0, 1]
    image = image / 255.0
    # Add channel dimension
    image = np.expand_dims(image, axis=0)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# Perform inference
def inference_with_onnx(image_path):
    input_data = preprocess_image(image_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    ort_inputs = {input_name: input_data}
    ort_outs = ort_session.run([output_name], ort_inputs)
    return ort_outs[0]

def decode_output(output_tensor, classes):
    # Apply softmax to obtain probabilities
    probabilities = np.exp(output_tensor) / np.sum(np.exp(output_tensor), axis=2, keepdims=True)
    # Find the index with the highest probability for each time step
    predicted_indices = np.argmax(probabilities, axis=2)
    # Convert indices to characters
    predicted_text = ''.join([classes[index] for index in predicted_indices[0]])
    return predicted_text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    filepath = sys.argv[1]
    output_tensor = inference_with_onnx(filepath)
    predicted_text_with_char_duplicates = decode_output(output_tensor, classes)
    predicted_text = decoded_answer_list_to_string(predicted_text_with_char_duplicates)

    print(predicted_text)
