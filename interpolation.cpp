#include<iostream>
#include<vector>
#include<stdio.h>
#include<algorithm>
#include<stdexcept>
#include<omp.h>
#include<cstdlib>
#include"opencv2/opencv.hpp"
#include<vector>
using namespace cv;
using namespace std;

void seq_data_copy(unsigned char *p, unsigned short *data, int size)
{
	int i = 0, j = 0;
	for (i, j; i < size; i += 4, j += 5)
	{
		data[i] = (p[j] << 2) + ((p[j + 4] >> 0) & 3);
		data[i + 1] = (p[j + 1] << 2) + ((p[j + 4] >> 2) & 3);
		data[i + 2] = (p[j + 2] << 2) + ((p[j + 4] >> 4) & 3);
		data[i + 3] = (p[j + 3] << 2) + ((p[j + 4] >> 6) & 3);
	}
}
void saveFile10bit(const char *filename, unsigned short * data, int dataWidth, int dataHeight)
{
	FILE *fp;
	fp = fopen(filename, "wb");
	fwrite(data, sizeof(unsigned short), dataWidth*dataHeight*3, fp);
	fclose(fp);
}
void serialInterpolation(unsigned short ** arr_red, unsigned short ** arr_blue, unsigned short ** arr_green, int width, int height)
{
	//interpolation of Red color serially
	for (int r = 0; r < height; r += 2)
		for (int c = 0; c < width; c += 2) {
			if (c < width - 2)
			{
				arr_red[r][c + 1] = (arr_red[r][c] + arr_red[r][c + 2]) / 2;
			}
		}
	for (int c = 0; c < width; c++)
		for (int r = 0; r < height; r += 2) {
			if (r < height - 2)
			{
				arr_red[r + 1][c] = (arr_red[r][c] + arr_red[r + 2][c]) / 2;
			}
		}
	//interpolation of Blue color serially
	for (int r = 1; r < height; r++)
	{
		for (int c = 1; c < width; c += 2)
		{
			if (c < width - 2)
			{
				arr_blue[r][c + 1] = (arr_blue[r][c] + arr_blue[r][c + 2]) / 2;
			}
			//arr_blue[r - 1][0] = arr_blue[r][1];
			//arr_blue[r][0] = arr_blue[r + 1][1];
		}
	}
	for (int c = 1; c < width; c++)
	{
		for (int r = 1; r < height; r += 2)
		{
			if (r < height - 2)
			{
				arr_blue[r + 1][c] = (arr_blue[r][c] + arr_blue[r + 2][c]) / 2;
			}
			arr_blue[0][c] = (arr_blue[1][c]);

		}
	}
	//interpolation of Green color serially
	for (int r = 1; r < height; r += 2)
	{
		for (int c = 1; c < width; c += 2)
		{
			if (c < width - 1 && c > 0 && r > 0 && r < height - 1)
			{
				arr_green[r][c] = (arr_green[r - 1][c] + arr_green[r][c - 1] + arr_green[r + 1][c] + arr_green[r][c + 1]) / 4;
			}
		}
	}
	for (int c = 0; c < width; c += 2)
	{
		for (int r = 0; r < height; r += 2)
		{
			if (c < width - 1 && c > 0 && r > 0 && r < height - 1)
			{
				arr_green[r][c] = (arr_green[r - 1][c] + arr_green[r][c - 1] + arr_green[r + 1][c] + arr_green[r][c + 1]) / 4;
			}
		}
	}
	for (int c = 0; c < width; c += 2)
	{
		for (int r = 0; r < height; r += 2)
		{
			if (c > 0 && r > 0 && c < width - 1 && r < height - 1)
			{
				arr_green[0][r] = (arr_green[0][r - 1] + arr_green[0][r + 1] + arr_green[1][r]) / 3;
			}
		}
	}
	for (int r = 0; r < height; r += 2)
	{
		for (int c = 0; c < width; c += 2)
		{
			if (c > 0 && r > 0 && c < width - 1 && r < height - 1)
			{
				arr_green[r][0] = (arr_green[r - 1][0] + arr_green[r + 1][0] + arr_green[r][1]) / 3;
			}
			if (c == 0 && r == 0)
			{
				arr_green[r][c] = (arr_green[r + 1][c] + arr_green[r][c + 1]) / 2;
			}
		}
	}
}// serial interpolation

//parallel interpolation technique
void interpolationRed(unsigned short ** arr, int width, int height)
{
#pragma omp parallel sections
	{
		#pragma omp section
		{
			//calculate column
			for (int r = 0; r < height; r += 2)
				for (int c = 0; c < width; c += 2) {
					if (c < width - 2)
					{
						arr[r][c + 1] = (arr[r][c] + arr[r][c + 2]) / 2;
					}
				}
		}
		#pragma omp section
		{
			//calculate row
			for (int c = 0; c < width; c++)
				for (int r = 0; r < height; r += 2) {
					if (r < height - 2)
					{
						arr[r + 1][c] = (arr[r][c] + arr[r + 2][c]) / 2;
					}
				}
		}
	}
}//
void interpolationBlue(unsigned short ** arr, int width, int height)
{//calculate column 
#pragma omp parallel sections
	{
		#pragma omp section
		{
			for (int r = 1; r < height; r++)
			{
				for (int c = 1; c < width; c += 2)
				{
					if (c < width - 2)
					{
						arr[r][c + 1] = (arr[r][c] + arr[r][c + 2]) / 2;
					}
					arr[r - 1][0] = arr[r][1];
				}
			}
		}

		//calculate row 
		#pragma omp section
		{
			for (int c = 1; c < width; c++)
			{
				for (int r = 1; r < height; r += 2)
				{
					if (r < height - 2)
					{
						arr[r + 1][c] = (arr[r][c] + arr[r + 2][c]) / 2;
					}
					arr[0][c] = (arr[1][c]);
				}
			}
		}

	}
}
void interpolationGreen(unsigned short ** arr, int width, int height)
{
#pragma omp parallel sections
	{
		#pragma omp section
		{
			for (int r = 1; r < height; r += 2)
			{
				for (int c = 1; c < width; c += 2)
				{
					if (c < width - 1 && c > 0 && r > 0 && r < height - 1)
					{
						arr[r][c] = (arr[r - 1][c] + arr[r][c - 1] + arr[r + 1][c] + arr[r][c + 1]) / 4;
					}
				}
			}
		}
		#pragma omp section
		{
			for (int c = 0; c < width; c += 2)
			{
				for (int r = 0; r < height; r += 2)
				{
					if (c < width - 1 && c > 0 && r > 0 && r < height - 1)
					{
						arr[r][c] = (arr[r - 1][c] + arr[r][c - 1] + arr[r + 1][c] + arr[r][c + 1]) / 4;
					}
				}
			}
		}
		#pragma omp section
		{
			for (int c = 0; c < width; c += 2)
			{
				for (int r = 0; r < height; r += 2)
				{
					if (c > 0 && r > 0 && c < width - 1 && r < height - 1)
					{
						arr[0][r] = (arr[0][r - 1] + arr[0][r + 1] + arr[1][r]) / 3;
					}
				}
			}
		}
		#pragma omp section
		{
			for (int r = 0; r < height; r += 2)
			{
				for (int c = 0; c < width; c += 2)
				{
					if (c > 0 && r > 0 && c < width - 1 && r < height - 1)
					{
						arr[r][0] = (arr[r - 1][0] + arr[r + 1][0] + arr[r][1]) / 3;
					}
					if (c == 0 && r == 0)
					{
						arr[r][c] = (arr[r + 1][c] + arr[r][c + 1]) / 2;
					}
				}
			}
		}
	}
}

int main()
{ 
	omp_set_num_threads(8);
	int height = 2448;
	int width = 3264;
	FILE *pFile;
	unsigned char* raw;
	//unsigned char* rgb;
	const char *filename;
	size_t result;

	pFile = fopen("raw.RAW", "rb");
	if (pFile == NULL)
	{
		fputs("File error", stderr);
		exit(1);
	}
	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);

	rewind(pFile);

	raw = (unsigned char*)malloc(sizeof(unsigned char)*lSize);
	if (raw == NULL)
	{
		fputs("Memory error", stderr);
		exit(2);
	}
	result = fread(raw, 1, lSize, pFile);
	if (result != lSize)
	{
		fputs("Reading error", stderr);
		exit(3);
	}
	//red blue green initializations
	unsigned short** red = (unsigned short**)malloc(sizeof(unsigned short*)*width);
	unsigned short** blue = (unsigned short**)malloc(sizeof(unsigned short*)*width);
	unsigned short** green = (unsigned short**)malloc(sizeof(unsigned short*)*width);

	for (int h = 0; h < height; h++)
	{
		red[h] = (unsigned short*)malloc(sizeof(unsigned short)*width);
		green[h] = (unsigned short*)malloc(sizeof(unsigned short)*width);
		blue[h] = (unsigned short*)malloc(sizeof(unsigned short)*width);
	}

	int counter = -1;
	unsigned short * data = (unsigned short*)malloc(sizeof(unsigned short)*height*width);//initial array of data 1-channel
	unsigned short * data_new = (unsigned short*)malloc(sizeof(unsigned short)*height*width*3);//1-D array to store final array 3-channel

	seq_data_copy(raw, data, height*width);//8-bit to 10-bit conversion

	//separating into RED, GREEN and BLUE 2D arrays to apply Bayer interpolation method
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			counter++;
			if ((c % 2 == 0) && (r % 2 == 0))
			{	red[r][c] = data[counter];}
			if ((c % 2) == 1 && (r % 2) == 0)
			{ green[r][c] = data[counter];}
			if ((c % 2) == 1 && (r % 2) == 1)
			{ blue[r][c] = data[counter];}
			if ((c % 2) == 0 && (r % 2) == 1)
			{green[r][c] = data[counter];}
		}
	}
	//creating variables to calculate processing time
	double start, end, procTime;
	double start_omp, end_omp, procTime_omp;
	//time calculations
	start_omp = omp_get_wtime();
	interpolationRed(red, width, height);
	interpolationGreen(green, width, height);
	interpolationBlue(blue, width, height);
	end_omp= omp_get_wtime();
	procTime_omp = (end_omp - start_omp);
	
	start = omp_get_wtime();
	serialInterpolation(red, blue, green, width, height);
	end = omp_get_wtime();
	procTime = (end - start);

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			cout << red[i][j] << " ";
		}
		cout << endl;
	}
	//converting 3 2D R,G,B arrays into one 1D array
	counter = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			data_new[counter] = blue[i][j];
			data_new[counter+1] = blue[i][j];
			data_new[counter+2] = blue[i][j];
			counter += 3;
		}
	}
	//converting into OpenCV Mat function to see image 
	Vec3w vec;
	Mat matrix = Mat(Size(width, height), CV_16UC3);
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			vec[0] = blue[r][c] * 64;
			vec[1] = green[r][c] * 64;
			vec[2] = red[r][c] * 64;
			matrix.at<Vec3w>(r, c) = vec;
		}
	}
	//imshow("mas", matrix);  //printing the image
	//imwrite("result.png", matrix);// saving image in png format
	saveFile10bit("said.raw", data_new, width, height);
	cout << "Processing time serial: " << procTime*1000 <<"msec" <<  endl; //calculating time for serial
	cout << "Processing time using omp: " << procTime_omp*1000 << "msec" << endl;//calculating time for parallel
	waitKey(0);
	system("pause");

	return 0;
}
