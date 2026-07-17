// Bayer (RGGB) demosaicing of a packed 10-bit RAW image, serial and OpenMP-parallel.
//
// Build (macOS, Homebrew libomp):
//   clang++ -O2 -std=c++11 interpolation.cpp -o interpolation \
//       -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
//       -L/opt/homebrew/opt/libomp/lib -lomp
// Build (Linux):
//   g++ -O2 -fopenmp interpolation.cpp -o interpolation
// Build with OpenCV display window: add -DUSE_OPENCV and the OpenCV
// include/link flags (on Visual Studio, define USE_OPENCV in project settings).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

using namespace std;

// Unpack 10-bit Bayer data: every 5 bytes hold 4 pixels
// (4 high bytes followed by 1 byte with the four 2-bit remainders).
void unpack10bit(const unsigned char *p, unsigned short *data, long pixelCount)
{
	for (long i = 0, j = 0; i < pixelCount; i += 4, j += 5)
	{
		data[i]     = (p[j]     << 2) | ((p[j + 4] >> 0) & 3);
		data[i + 1] = (p[j + 1] << 2) | ((p[j + 4] >> 2) & 3);
		data[i + 2] = (p[j + 2] << 2) | ((p[j + 4] >> 4) & 3);
		data[i + 3] = (p[j + 3] << 2) | ((p[j + 4] >> 6) & 3);
	}
}

// Allocate a height x width 2D array backed by one contiguous zeroed buffer.
unsigned short **alloc2d(int height, int width)
{
	unsigned short **rows = (unsigned short **)malloc(sizeof(unsigned short *) * height);
	unsigned short *buf = (unsigned short *)calloc((size_t)height * width, sizeof(unsigned short));
	if (rows == NULL || buf == NULL)
	{
		fputs("Memory error\n", stderr);
		exit(2);
	}
	for (int r = 0; r < height; r++)
		rows[r] = buf + (size_t)r * width;
	return rows;
}

void free2d(unsigned short **arr)
{
	free(arr[0]);
	free(arr);
}

void copy2d(unsigned short **dst, unsigned short **src, int height, int width)
{
	memcpy(dst[0], src[0], (size_t)height * width * sizeof(unsigned short));
}

// Mirror an out-of-range index back inside [0, n): -1 -> 1, n -> n-2.
// Keeps the index parity, so a reflected neighbour is still a known pixel.
static inline int reflect(int i, int n)
{
	return i < 0 ? 1 : (i >= n ? n - 2 : i);
}

// Red is known at (even row, even col).
// Pass 1 fills odd columns of even rows, pass 2 fills all odd rows.
// The two passes must run one after the other (pass 2 reads pass 1's output),
// but each pass is independent across rows, so `parallel for` is race-free.
void interpolateRed(unsigned short **a, int width, int height, int parallel)
{
#pragma omp parallel for if(parallel)
	for (int r = 0; r < height; r += 2)
	{
		for (int c = 1; c < width - 1; c += 2)
			a[r][c] = (a[r][c - 1] + a[r][c + 1]) / 2;
		if (width % 2 == 0)
			a[r][width - 1] = a[r][width - 2];
	}
#pragma omp parallel for if(parallel)
	for (int r = 1; r < height - 1; r += 2)
		for (int c = 0; c < width; c++)
			a[r][c] = (a[r - 1][c] + a[r + 1][c]) / 2;
	if (height % 2 == 0)
		memcpy(a[height - 1], a[height - 2], (size_t)width * sizeof(unsigned short));
}

// Blue is known at (odd row, odd col); same two-pass scheme as red,
// with the first row/column filled by replication.
void interpolateBlue(unsigned short **a, int width, int height, int parallel)
{
#pragma omp parallel for if(parallel)
	for (int r = 1; r < height; r += 2)
	{
		for (int c = 2; c < width; c += 2)
			a[r][c] = (a[r][c - 1] + a[r][c + 1]) / 2;
		a[r][0] = a[r][1];
	}
#pragma omp parallel for if(parallel)
	for (int r = 2; r < height; r += 2)
		for (int c = 0; c < width; c++)
			a[r][c] = (a[r - 1][c] + a[r + 1][c]) / 2;
	memcpy(a[0], a[1], (size_t)width * sizeof(unsigned short));
}

// Green is known where (row + col) is odd. Every missing pixel's four
// neighbours are known greens, so one pass over the missing positions
// reads only original values — safe to parallelise over rows.
void interpolateGreen(unsigned short **a, int width, int height, int parallel)
{
#pragma omp parallel for if(parallel)
	for (int r = 0; r < height; r++)
	{
		for (int c = (r % 2 == 0) ? 0 : 1; c < width; c += 2)
		{
			int up = reflect(r - 1, height), down = reflect(r + 1, height);
			int left = reflect(c - 1, width), right = reflect(c + 1, width);
			a[r][c] = (a[up][c] + a[down][c] + a[r][left] + a[r][right]) / 4;
		}
	}
}

void interpolateAll(unsigned short **red, unsigned short **green, unsigned short **blue,
                    int width, int height, int parallel)
{
	interpolateRed(red, width, height, parallel);
	interpolateGreen(green, width, height, parallel);
	interpolateBlue(blue, width, height, parallel);
}

// Save interleaved B,G,R 16-bit samples (10-bit values) as raw binary.
void saveFile10bit(const char *filename, const unsigned short *data, int width, int height)
{
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL)
	{
		perror(filename);
		return;
	}
	fwrite(data, sizeof(unsigned short), (size_t)width * height * 3, fp);
	fclose(fp);
}

// Save an 8-bit PPM (viewable without OpenCV); 10-bit values are scaled down.
void savePPM(const char *filename, unsigned short **red, unsigned short **green,
             unsigned short **blue, int width, int height)
{
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL)
	{
		perror(filename);
		return;
	}
	fprintf(fp, "P6\n%d %d\n255\n", width, height);
	unsigned char *row = (unsigned char *)malloc((size_t)width * 3);
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			row[c * 3]     = (unsigned char)(red[r][c] >> 2);
			row[c * 3 + 1] = (unsigned char)(green[r][c] >> 2);
			row[c * 3 + 2] = (unsigned char)(blue[r][c] >> 2);
		}
		fwrite(row, 1, (size_t)width * 3, fp);
	}
	free(row);
	fclose(fp);
}

int main()
{
	const int height = 2448;
	const int width = 3264;
	const long pixelCount = (long)height * width;

	FILE *pFile = fopen("raw.RAW", "rb");
	if (pFile == NULL)
	{
		fputs("File error: cannot open raw.RAW\n", stderr);
		return 1;
	}
	fseek(pFile, 0, SEEK_END);
	long lSize = ftell(pFile);
	rewind(pFile);

	if (lSize != pixelCount * 5 / 4)
	{
		fprintf(stderr, "Unexpected file size %ld (expected %ld for %dx%d packed 10-bit)\n",
		        lSize, pixelCount * 5 / 4, width, height);
		fclose(pFile);
		return 1;
	}

	unsigned char *raw = (unsigned char *)malloc((size_t)lSize);
	unsigned short *data = (unsigned short *)malloc(sizeof(unsigned short) * pixelCount);
	if (raw == NULL || data == NULL)
	{
		fputs("Memory error\n", stderr);
		return 2;
	}
	if (fread(raw, 1, (size_t)lSize, pFile) != (size_t)lSize)
	{
		fputs("Reading error\n", stderr);
		return 3;
	}
	fclose(pFile);

	unpack10bit(raw, data, pixelCount);
	free(raw);

	// Split the RGGB mosaic into per-channel planes (missing pixels stay 0).
	unsigned short **red = alloc2d(height, width);
	unsigned short **green = alloc2d(height, width);
	unsigned short **blue = alloc2d(height, width);
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			unsigned short v = data[(long)r * width + c];
			if (r % 2 == 0 && c % 2 == 0)
				red[r][c] = v;
			else if (r % 2 == 1 && c % 2 == 1)
				blue[r][c] = v;
			else
				green[r][c] = v;
		}
	}
	free(data);

	// Second copy of the fresh (un-interpolated) planes so the serial and
	// parallel runs each start from the same input.
	unsigned short **red2 = alloc2d(height, width);
	unsigned short **green2 = alloc2d(height, width);
	unsigned short **blue2 = alloc2d(height, width);
	copy2d(red2, red, height, width);
	copy2d(green2, green, height, width);
	copy2d(blue2, blue, height, width);

	double start = omp_get_wtime();
	interpolateAll(red, green, blue, width, height, 0);
	double procTime = omp_get_wtime() - start;

	start = omp_get_wtime();
	interpolateAll(red2, green2, blue2, width, height, 1);
	double procTime_omp = omp_get_wtime() - start;

	bool identical =
	    memcmp(red[0], red2[0], (size_t)pixelCount * sizeof(unsigned short)) == 0 &&
	    memcmp(green[0], green2[0], (size_t)pixelCount * sizeof(unsigned short)) == 0 &&
	    memcmp(blue[0], blue2[0], (size_t)pixelCount * sizeof(unsigned short)) == 0;

	cout << "OpenMP threads: " << omp_get_max_threads() << endl;
	cout << "Processing time serial: " << procTime * 1000 << " msec" << endl;
	cout << "Processing time using omp: " << procTime_omp * 1000 << " msec" << endl;
	cout << "Serial and parallel results " << (identical ? "match" : "DIFFER!") << endl;

	// Interleave into one B,G,R 1-D array and save.
	unsigned short *data_new = (unsigned short *)malloc(sizeof(unsigned short) * pixelCount * 3);
	if (data_new == NULL)
	{
		fputs("Memory error\n", stderr);
		return 2;
	}
	long counter = 0;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			data_new[counter]     = blue[r][c];
			data_new[counter + 1] = green[r][c];
			data_new[counter + 2] = red[r][c];
			counter += 3;
		}
	}
	saveFile10bit("said.raw", data_new, width, height);
	savePPM("result.ppm", red, green, blue, width, height);
	free(data_new);

#ifdef USE_OPENCV
	cv::Mat matrix(cv::Size(width, height), CV_16UC3);
	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++)
			matrix.at<cv::Vec3w>(r, c) =
			    cv::Vec3w(blue[r][c] * 64, green[r][c] * 64, red[r][c] * 64);
	cv::imshow("result", matrix);
	cv::imwrite("result.png", matrix);
	cv::waitKey(0);
#endif

	free2d(red);
	free2d(green);
	free2d(blue);
	free2d(red2);
	free2d(green2);
	free2d(blue2);
	return 0;
}
