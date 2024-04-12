import torch
import torchvision.transforms as transforms

from models.crnn import CRNN
from static_variables import classes

# Define input shape
image_width = 250
image_height = 60

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# Initialize the model
model = CRNN(
    resolution=(image_width, image_height),
    dims=256,
    num_chars=len(classes) - 1,
    use_attention=True,
    use_ctc=True,
    grayscale=True,
)

# Load the trained weights
weights_path = "logs-2024-02-18-data-5k-no-errors-epochs-20/crnn.pth"
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 1, image_height, image_width)  # Assuming single channel (grayscale) input

# Convert the model to ONNX format
onnx_path = "crnn.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False, input_names=['input'], output_names=['output'], opset_version=11)
