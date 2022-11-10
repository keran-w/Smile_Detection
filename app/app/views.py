import os
import json
import shutil

from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseNotFound, HttpResponse, HttpResponseRedirect
from django.core.files.storage import FileSystemStorage

from .settings import MEDIA_ROOT

def index(request):
    context = {}
    return render(request, 'index.html', context)