import torch
import numpy as np
from PIL import Image

from models.crnn import CRNN
from static_variables import classes
from utils.model_decoders import decode_predictions, decode_padded_predictions
from torchvision import transforms
import sys

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def inference(image_path):
    # Hardcoded resize
    image = Image.open(image_path).convert("RGB")
    image = image.resize((250, 60), resample=Image.BILINEAR)
    image = transform(image)
    image = np.array(image)

    # ImageNet mean and std (not required, but if you trained with, keep it)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    # image = aug(image=image)["image"]
    # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = image[None, ...]
    image = torch.from_numpy(image)
    if str(device) == "cuda":
        image = image.cuda()
    image = image.float()
    with torch.no_grad():
        preds, _ = model(image)

    if model.use_ctc:
        answer = decode_predictions(preds, classes)
        answer = decoded_answer_list_to_string(answer)
    else:
        answer = decode_padded_predictions(preds, classes)
    return answer

def decoded_answer_list_to_string(answer):
    # collapse each sequence of chars (length up to 3) that are siblings and duplicates into one
    for i in range(len(answer) - 1, 0, -1):
        if answer[i] == answer[i - 1]:
            answer.pop(i)
    answer = "".join(answer)
    return answer

if __name__ == "__main__":
    # Setup model and load weights
    model = CRNN(
        resolution=(250, 60),
        dims=256,
        num_chars=len(classes) - 1,
        use_attention=True,
        use_ctc=True,
        grayscale=True,
    )
    device = torch.device("cpu")
    model.to(device)
    model.load_state_dict(torch.load("./logs/crnn.pth"))
    model.eval()
    # get file path from cli args
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    filepath = sys.argv[1]
    answer = inference(filepath)
    print(f"text: {answer}")
