# StyleTransfer
This is a fun application I build to practice about Deep Learning using Django.
The application is about Style Transfer learning where you input the content image and style image, later it will produce new image which is based the content from the first image with style from style image.
I built this application in 3 days and I only focus on the Deep Learning model. So the website is really simple.

I have two version.
One is the model I built based on the Coursera Deep Learning's course assignment (You can see code at transfer.py and helper.py)
It took quite a long time to produce the styled image. So I moved to next one.

The another is I use the pretrained model downloaded from tensorflow hub and it outputs styled image much faster (fast_transfer.py, fast_helper.py).

This application is not a big deal but I've learned so many things while making it such as problems related to heroku, how to store uploaded images by users at Amazon S3 or how to make model smaller in order to be able deploy it to heroku, because heroku only allows people upload application with maximum size if 500M,...
