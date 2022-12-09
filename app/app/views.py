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
from .settings import MEDIA_ROOT


# Build model structure
from torch import nn
from torch.nn import functional as F
import numpy as np

import cv2
import torch
import albumentations as A



def index(request):
    context = {}
    return render(request, 'index.html', context)

@csrf_exempt 
def detect(request):
    result = json.load(open('cache.json', 'r'))['output']
    if result:
        cache = {}
        cache['output'] = []
        json.dump(cache, open('cache.json', 'w'))
        if result[0] > 0.5:
            return JsonResponse({"result": "Not Smiling", "prob": result[0]})
        else:
            return JsonResponse({"result": "Smiling", "prob": 1 - result[0]})
    data = request.POST.getlist('data[]')
    # print(data)
    # os.system("")
    data = np.reshape(data, (64, 64))
    # np.save('cache', data)  
    cache = {}
    cache['data'] = data.tolist()
    cache['output'] = []
    
    json.dump(cache, open('cache.json', 'w'))
    
    import os
    import time
    start = time.time()
    os.system('python inference/inference.py')
    print(time.time() - start)
    # sleep()
    
    
    # os.system('pyto...')
    # non_smile, smile = inference(r"saved_models\vgg_inference.pt")
    return JsonResponse({"result": "Waiting"})
