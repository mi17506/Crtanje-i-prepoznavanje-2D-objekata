import os
import sys
import random
import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from neuralnet import *
from nnmath import *
from genetics import GeneticAlgorithm, GAKill
import pickle
import cv2

from PIL import ImageTk, Image, ImageDraw
import PIL
from Tkinter import *


def read_data(path):
	data = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for dirname in dirnames:
			for f in os.listdir(dirpath + dirname):
				try:
				    img = np.ravel(misc.imread(dirpath + dirname + '/' + f, flatten=True))/255
				    data.append((dirname, img))
				except:
					pass
	return data

def main(argv):
	np.seterr(over='ignore')

	
	targets = np.array(['rectangle', 'circle', 'triangle'])

	if argv[1] == 'train':
		# Proveravanje ulaznih argumenata
		if len(argv) < 5:
			print ("Usage: python shape.py train <GA-epochs> <SGD-epochs> <visFlag>")
			sys.exit()

		# Ucitavanje trening podataka
		training_data = read_data('training_data/')
		test_data = read_data('test_data/')

		# Nasumicno mesanje podataka
		random.shuffle(training_data)

		# Pravljenje GA od neuronske mreze
		img_len = len(training_data[0][1])
		ga = GeneticAlgorithm(epochs = int(argv[2]),
								mutation_rate = 0.01,
								data = training_data,
								targets = targets,
								obj = NeuralNet,
								args = [img_len, 10, 4, 3])

		# Kreiranje prve populacije
		print ("Creating population...")
		ga.populate(200)

		print ("Initiating GA heuristic approach...")

		# Pocetak evolucije
		errors = []
		while ga.evolve():
			try:
				ga.evaluate()
				ga.crossover()
				ga.epoch += 1

				
				errors.append(ga.error)
				print ("error: " + str(ga.error))
			except GAKill as e:
				break

		vis = bool(int(argv[4]))


		print ("--------------------------------------------------------------\n")

		nn = ga.fittest()
		epochs = int(argv[3])
		if epochs:
			print ("Initiating Gradient Descent optimization...")
			try:
				nn.gradient_descent(training_data, targets, epochs, test_data=test_data, vis=vis)
			except GAKill as e:
				print (e.message)

		nn.save("neuralnet.pkt")
		print ("Done!")


	elif argv[1] == "predict":
		# Proveravanje ulaznih argumenata
		if len(argv) < 2:
		    print ("Usage: python shape.py predict")
		    sys.exit()

		# Ucitavanje nacrtane slike
		img = np.ravel(misc.imread("image.jpg", flatten=True))/255

		# Izgradnja neuronske mreze iz fajla
		nn = NeuralNet([], build=False)
		nn.load("neuralnet.pkt")

		
		activations, zs = nn.feed_forward(img)
                probability = max(activations[-1])
		print ("Shape is: "+targets[np.argmax(activations[-1])] )
		print ("With probability: ")
		print ( probability )
		
	elif argv[1] == "validate":
		test_data = read_data('test_data/')

		nn = NeuralNet([], build=False)
		nn.load("neuralnet.pkt")

		accuracy = nn.validate(targets, test_data)
		print "Accuracy: " + str(accuracy)
		
	elif argv[1] == "draw":
		#Proveravanje ulaznih argumenata


                #Inicijalizovanje parametara table za crtanje
                width = 100
                height = 100
                center = height//2
                white = (255, 255, 255)
                green = (0,128,0)

                #Funkcija cuvanja nacrtane slike 
                def save():
                    filename = "image.jpg"
                    image1.save(filename)
                    exit()

                #Crtanje oblika
                def paint(event):
                
                    x1, y1 = (event.x - 1), (event.y - 1)
                    x2, y2 = (event.x + 1), (event.y + 1)
                    cv.create_oval(x1, y1, x2, y2, fill="black",width=10)
                    draw.line([x1, y1, x2, y2],fill="black",width=10)

                root = Tk()

                cv = Canvas(root, width=width, height=height, bg='white')
                cv.pack()

                image1 = PIL.Image.new("RGB", (width, height), white)
                draw = ImageDraw.Draw(image1)


                cv.pack(expand=YES, fill=BOTH)
                cv.bind("<B1-Motion>", paint)

                
                button=Button(text="save",command=save)
                button.pack()
                root.mainloop()
                

	else:
		print ("ERROR: Unknown command " + argv[1])

# Postavljanje upravljaca za obradu greske nastale usled nasilnog prekidanja programa
def signal_handler(signal, frame):
	raise(GAKill("\nAborting Search..."))

if __name__ == "__main__":
	signal.signal(signal.SIGINT, signal_handler)
	main(sys.argv)
