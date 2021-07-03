import csv
from PIL import Image
import os
import numpy as np
from tensorflow import keras
loaded_model = keras.models.load_model('digits_recognition_cnn.h5')

with open('AIMLC_HackTheSummer_2.csv', mode='w') as opfile:
    or_writer = csv.writer(opfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    with open('annotations.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            imageName = row[0]
            expression_type = row[1]
            path = os.path.join("data", imageName)
            if(os.path.exists(path)):
                im = Image.open(path, "r")
                width, height = im.size
                # box1 is 0  to width/3 
                box1 = (0, 0, width/3, height)
                # box2 is width/3 to 2*width/3
                box2 = (width/3, 0, 2*width/3, height)
                # box3 is 2*width/3 to width
                box3 = (2*width/3, 0, width, height)
                im1 = im.crop(box1)
                im2 = im.crop(box2)
                im3 = im.crop(box3)
                
                im1 = im1.resize((28,28))
                np.array(im1)
                im2 = im2.resize((28,28))
                np.array(im2)
                im3 = im3.resize((28,28))
                np.array(im3)
                
                pix1 = list(list(im1.getdata()))
                pixel1 = np.array(pix1)
                pixel1 = pixel1/255
                pixel1 = pixel1.reshape(1,28,28,1)


                pix2 = list(list(im2.getdata()))
                pixel2 = np.array(pix2)
                pixel2 = pixel2/255
                pixel2 = pixel2.reshape(1,28,28,1)
                
                pix3 = list(list(im3.getdata()))
                pixel3 = np.array(pix3)
                pixel3 = pixel3/255
                pixel3 = pixel3.reshape(1,28,28,1)
                
                dig1 = np.argmax(loaded_model.predict(pixel1), axis =1)[0]
                dig1 = int(dig1)
                dig2 = np.argmax(loaded_model.predict(pixel2), axis =1)[0]
                dig2 = int(dig2)
                dig3 = np.argmax(loaded_model.predict(pixel3), axis =1)[0]
                dig3 = int(dig3)
                
                if(dig1>9 and dig2<10 and dig3<10):
                    if(dig1==10):
                        or_writer.writerow([imgname,dig2+dig3])
                    elif(dig1==11):
                        or_writer.writerow([imgname,dig2-dig3])
                    elif(dig1==12):
                        or_writer.writerow([imgname,dig2*dig3])
                    elif(dig1==13):
                        or_writer.writerow([imgname,dig2//dig3])
                elif(dig1<10 and dig2>9 and dig3<10):
                    if(dig2==10):
                        or_writer.writerow([imgname,dig1+dig3])
                    elif(dig2==11):
                        or_writer.writerow([imgname,dig1-dig3])
                    elif(dig2==12):
                        or_writer.writerow([imgname,dig1*dig3])
                    elif(dig2==13):
                        or_writer.writerow([imgname,dig1//dig3])
                elif(dig1<10 and dig2<10 and dig3>9):
                    if(dig3==10):
                        or_writer.writerow([imgname,dig1+dig2])
                    elif(dig3==11):
                        or_writer.writerow([imgname,dig1-dig2])
                    elif(dig3==12):
                        or_writer.writerow([imgname,dig1*dig2])
                    elif(dig==13):
                        or_writer.writerow([imgname,dig1//dig2])