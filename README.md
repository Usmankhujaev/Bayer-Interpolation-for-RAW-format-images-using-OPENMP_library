# Bayer-Interpolation-for-RAW-format-images-
The program that creates a Bayer filter mosaic color filter array for arranging RGB color filters.
What is Bayer filter?
It is a color filter array for arranging Red, Green and Blue color filters. 
Invented by Bryce Bayer in 1976
The Bayer filter is almost universal on consumer digital cameras

![Alt text](/images/bayer.png)

Tasks performed
1) Load the .raw format image 
2) Read the file and store in a one-dimensional array
3) Allocate the memory space for image
4) Convert 8-bit image values to 10-bit
5) Separate each R,G and B channels
6) Interpolate each channel
7) Interpolate using OpenMP
8) Convert to OpenCV Mat format
9) Save the image


Channel separation

![Alt text](/images/bayer_3.jpg)

Missing pixel values is filled by averaging the values of each channel 



Averaging formula for red and blue channels
![Alt text](/images/bayer_4.PNG)

Averaging formula for green channel
![Alt text](/images/bayer_5.png)



For detailed information check this file
