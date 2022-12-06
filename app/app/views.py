import os
import json
import shutil

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseNotFound, HttpResponse, HttpResponseRedirect, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt


import cv2
import torch
import albumentations as A

from model import *

def index(request):
    context = {}
    return render(request, 'index.html', context)

@csrf_exempt 
def detect(request):
    data = request.POST.getlist('data[]')
    print(data)
    
    return JsonResponse({"result": "Smiling"})


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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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


def inference(image_arr, model_path, device='cpu'):
    # Load image from path
    transform = A.Compose([
        A.Resize(64, 64),
        A.Normalize(0.449, 0.226)
    ])
    image = image_transform(image_arr, transform)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)

    # Load the inference model
    inference_model = load_model(model_path, device)

    # Get outputs and normalize
    return torch.sigmoid(inference_model(image))[0].tolist()
