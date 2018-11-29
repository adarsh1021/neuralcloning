from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
import base64
# from .celery_test import test

# from clone_script import clone

def index(request):
	return render(request, 'clone/index.html')

def results(request):
	return render(request, 'clone/results.html')

def login(request):
	return render(request, 'clone/login.html')

@csrf_exempt
def upload(request):
	if request.method == 'POST':
		data = request.POST["base64_img"]
		format, imgstr = data.split(';base64,')
		ext = '.'+format.split('/')[-1]
		# print(ext)
		directory = "STEPS/0"
		filename = directory + ".jpg"
		with open(filename, "wb") as fh:
		    fh.write(base64.b64decode(imgstr))
		# clone.delay()
		return HttpResponse("success")
	return HttpResponse("fail")