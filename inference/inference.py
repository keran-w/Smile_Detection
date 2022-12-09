import cv2
import torch
import albumentations as A
import json

from model import *

# Global variable to check if a model is used for the first time
MODELWARMUP = True

# Display image from path
def display_image(image_path):
    from PIL import Image
    return Image.open(image_path)

# Read an image to array
def get_image_arr(image_pth):
    return cv2.imread(image_pth)

# Transform an image to fit the inference model
def image_transform(image, transform):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    import numpy as np
    image = np.array(image).astype('int')
    return transform(image=image)['image']

# Load the inference model from path
def load_model(model_path, device):
    # Load model
    inference_model = torch.load(model_path)
    inference_model.eval()
    inference_model.to(device)

    # Warm up
    global MODELWARMUP
    if MODELWARMUP:
        inference_model(torch.randn(1, 1, 64, 64).to(device))
        MODELWARMUP = False
    return inference_model


def inference(model_path, device='cpu'):
    # Load image from path
    transform = A.Compose([
        A.Resize(64, 64),
        A.Normalize(0.449, 0.226)
    ])
    import json
    image = json.load(open('cache.json', 'r'))['data']
    
    # image = get_image_arr(image)
    image = image_transform(image, transform)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    # Load the inference model
    inference_model = load_model(model_path, device)

    # Get outputs and normalize
    return torch.sigmoid(inference_model(image))[0].tolist()



if __name__ == '__main__':
    
    
    # Define label map {id -> name}
    label2id = {'non-smile': 0, 'smile': 1}
    id2label = {v: k for k, v in label2id.items()}
    labels = list(label2id.keys())

    # Set paths and predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = './static/image/1.jpg'
    model_path = './saved_models/vgg_inference.pt'
    probs = inference(model_path, device)

    # Display outputs
    print(probs)
    probs_map = {k: v for k, v in zip(labels, probs)}
    print(probs_map)
    cache = {}
    cache['output'] = [probs[0]]
    json.dump(cache, open('cache.json', 'w'))
