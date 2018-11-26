from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
import base64

def index(request):
	return render(request, 'clone/index.html')

@csrf_exempt
def upload(request):
	if request.method == 'POST':
		data = request.POST["base64_img"]
		format, imgstr = data.split(';base64,')
		ext = '.'+format.split('/')[-1] 
		directory = "data/"+request.POST['class'][0]
		filename = str(len(os.listdir(directory))+1) + ext
		with open(directory+"/"+filename, "wb") as fh:
		    fh.write(base64.b64decode(imgstr))
		return HttpResponse("success")
	return HttpResponse("fail")