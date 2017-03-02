from flask import Flask, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image

import os
os.environ["GLOG_minloglevel"] = "1"
import caffe

caffe.set_mode_cpu()
# read labels
with open('words1000.txt', 'r') as f:
	words=list(f)
# load model
net = caffe.Net('deploy.prototxt', 1, weights='resnet50_cvgj_iter_112099.caffemodel')
print '--------------------------------------------------------------------------------'

# web server parameters
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['JPG', 'JPEG', 'jpg', 'jpeg'])

# presentation
header = '<!doctype html><html><body>'
html='<form action="/predict" method="post" enctype="multipart/form-data"><input type=file name="file"><br/><input type="submit"></form>'
footer='</body></html>'

################################ Flask server #######################
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize(filename):
	im = Image.open(filename)
	im2 = im.resize((256,256))
	im2.save(filename)

# call caffe to make a prediction on the uploaded file
@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		# save file to disk
		if 'file' not in request.files:
			print('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			print('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)	
		else:
			print('File not allowed')
			return redirect(request.url)		
		# read image and resize it
		resize(file_path)
		im = np.array(Image.open(file_path))
		#crop image
		im2 = caffe.io.oversample((im,),(224,224))
		#take first crop
		net.blobs['data'].data[...] = np.transpose(im2[0:1,:,:,:], (0,3,1,2))
		# forward propagation
		net.forward()
		#compute output
		softmax_probabilities = net.blobs['predict'].data[0,:]
		#compute top 5 sorted in ascending probability order
		ranks = np.argpartition(softmax_probabilities, -5)[-5:]
		ranks = ranks[np.argsort(softmax_probabilities[ranks])]
		# print top 5
		print ranks
		response=header
		response += '<img src="/image/' + filename + '"></img><br>'
		for rank in ranks:
			line = "output rank {} - proba : {:05.2f} % - label  {} ".format(rank,100*softmax_probabilities[rank],words[rank].rstrip())
			print line
			response += line
			response += "<br/>"
		return response + '<br>' + html + footer
	else:
		return header + html + footer

# display the uploaded image
@app.route('/image/<filename>',methods=['GET'])
def image(filename):
	return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

