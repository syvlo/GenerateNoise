#include <iostream>
#include <chrono>
#include <random>

#include <cstring>
#include <cstdlib>

#include <cv.h>
#include <highgui.h>

#include <stdio.h>

#include "ImwHelper.hh"

#define NB_BINS (200)
#define HEIGHT (300)
void
printHelp()
{
    std::cout << "Generate noise on images." << std::endl
	      << "Usage:" << std::endl
	      << "./GenerateNoise [-radar] gaussian stddev input output [-check]" << std::endl
	      << "./GenerateNoise [-radar] rayleigh stddev input output [-check]" << std::endl
	      << "./GenerateNoise [-radar] nakagami L input output [-check]" << std::endl;
}

void
checkDistrib(const cv::Mat noise, unsigned minIm, unsigned maxIm, bool additive)
{
    unsigned width = noise.size().width;
    unsigned height = noise.size().height;
    double nbPix = width * height;


    //Retrieve mean, min and max.
    double max = 0;
    double min = 4242;

    double mean = 0;

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    mean += noise.at<float>(i, j);
	    if (noise.at<float>(i, j) > max)
		max = noise.at<float>(i, j);
	    if (noise.at<float>(i, j) < min)
		min = noise.at<float>(i, j);
	}
    mean /= nbPix;

    std::cout << "Mean = " << mean << std::endl;

    //Retrieve stddev and fill bins.
    float std = 0;
    std::vector<unsigned> bins (NB_BINS, 0);
    float interval = (max - min) / NB_BINS;

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    std += (noise.at<float>(i, j) - mean) * (noise.at<float>(i, j) - mean);
	    ++bins[(noise.at<float>(i, j) - min) / interval];
	}
    std /= nbPix;
    std = sqrt(std);

    std::cout << "Standart deviation = " << std << std::endl;
    std::cout << "Coefficient of variation = " << std / mean << std::endl;

    if ((additive && minIm + min < 0) || (!additive && minIm * min < 0))
	std::cout << "Warning: some values could be < 0." << std::endl;
    if (maxIm > 0)//in radar mode, maxIm == 0 
    {
	if ((additive && maxIm + max > 255) || (!additive && maxIm * max > 255))
	    std::cout << "Warning: some values could be > 255." << std::endl;
    }

    unsigned maxBin = 0;
    for (unsigned i = 0; i < NB_BINS; ++i)
	if (bins[i] > maxBin)
	    maxBin = bins[i];

    cv::Mat histImage (HEIGHT + 10, NB_BINS, CV_8UC3, cv::Scalar(255, 255, 255));

    for (unsigned i = 0; i < NB_BINS - 1; ++i)
	cv::line(histImage, cv::Point(i, HEIGHT - bins[i] * HEIGHT / maxBin), cv::Point(i + 1, HEIGHT - bins[i + 1] * HEIGHT / maxBin), cv::Scalar(0, 0, 0));

    char buffer[10];
    sprintf(buffer,"%d",(int)mean);
    cv::putText(histImage, std::string(buffer), cv::Point((mean  - min) / interval, HEIGHT + 7), 0, 0.3, cv::Scalar(255, 0, 0));

    sprintf(buffer,"%d",(int)min);
    cv::putText(histImage, std::string(buffer), cv::Point(1, HEIGHT + 7), 0, 0.3, cv::Scalar(255, 0, 0));

    sprintf(buffer,"%d",(int)max);
    cv::putText(histImage, std::string(buffer), cv::Point((max - min) / interval - 20, HEIGHT + 7), 0, 0.3, cv::Scalar(255, 0, 0));

    sprintf(buffer,"%d",(int)maxBin);
    cv::putText(histImage, std::string(buffer), cv::Point(1, 10), 0, 0.3, cv::Scalar(255, 0, 0), 1, 8, false);

    cv::imwrite("hist.png", histImage);
}

cv::Mat
gaussian(float stddev, const cv::Mat input, bool check)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution (0.0,stddev);

    unsigned width = input.size().width;
    unsigned height = input.size().height;

    cv::Mat output(input.size(), input.type());

    cv::Mat noise(input.size(), CV_32F);
    unsigned min = 42424242;
    unsigned max = 0;

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    float noiseValue = distribution(generator);
	    if (check)
	    {
		noise.at<float>(i, j) = noiseValue;
		if (input.type() == CV_8U)
		{
		    if (min > input.at<unsigned char>(i, j))
			min = input.at<unsigned char>(i, j);
		    if (max < input.at<unsigned char>(i, j))
			max = input.at<unsigned char>(i, j);
		}
		else
		{
		    if (min > input.at<unsigned short>(i, j))
			min = input.at<unsigned short>(i, j);
		}
	    }
	    float value;
	    if (input.type() == CV_8U)
		value = input.at<unsigned char>(i, j) + noiseValue;
	    else
		value = input.at<unsigned short>(i, j) + noiseValue;
	    if (input.type() == CV_8U)
	    {
		unsigned char truncated;
		if (value < 0)
		    truncated = 0;
		else
		    if (value > 255)
			truncated = 255;
		    else
			truncated = value;
		output.at<unsigned char>(i, j) = truncated;
	    }
	    else
	    {
		unsigned short truncated;
		if (value < 0)
		    truncated = 0;
		else
		    truncated = value;
		output.at<unsigned short>(i, j) = truncated;
	    }
	}

    if (check)
	checkDistrib(noise, min, max, true);

    return output;
}

cv::Mat
rayleigh(float sigma, const cv::Mat input, bool check)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<float> distribution(0.0,1.0);

    unsigned width = input.size().width;
    unsigned height = input.size().height;

    cv::Mat output(input.size(), input.type());

    cv::Mat noise(input.size(), CV_32F);
    unsigned min = 42424242;
    unsigned max = 0;

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    float noiseValue = sigma * sqrt(-2 * log(distribution(generator)));
	    if (check)
	    {
		noise.at<float>(i, j) = noiseValue;
		if (input.type() == CV_8U)
		{
		    if (min > input.at<unsigned char>(i, j))
			min = input.at<unsigned char>(i, j);
		    if (max < input.at<unsigned char>(i, j))
			max = input.at<unsigned char>(i, j);
		}
		else
		{
		    if (min > input.at<unsigned short>(i, j))
			min = input.at<unsigned short>(i, j);
		}
	    }
	    float value;
	    if (input.type() == CV_8U)
		value = (float)input.at<unsigned char>(i, j) * noiseValue;
	    else
		value = (float)input.at<unsigned short>(i, j) * noiseValue;
	    if (input.type() == CV_8U)
	    {
		unsigned char truncated;
		if (value < 0)
		    truncated = 0;
		else
		    if (value > 255)
			truncated = 255;
		    else
			truncated = value;
		output.at<unsigned char>(i, j) = truncated;
	    }
	    else
	    {
		unsigned short truncated;
		if (value < 0)
		    truncated = 0;
		else
		    truncated = value;
		output.at<unsigned short>(i, j) = truncated;
	    }
	}

    if (check)
	checkDistrib(noise, min, max, false);

    return output;
}

cv::Mat
nakagami(int L, const cv::Mat input, bool check)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution (0.0,1.0);

    unsigned width = input.size().width;
    unsigned height = input.size().height;

    cv::Mat output(input.size(), input.type());

    cv::Mat noise = cv::Mat::zeros(input.size(), CV_32F);
    unsigned min = 42424242;
    unsigned max = 0;

    for (unsigned k = 0; k < L; ++k)
	for (unsigned i = 0; i < height; ++i)
	    for (unsigned j = 0; j < width; ++j)
	    {
		float a = distribution(generator);
		float b = distribution(generator);
		noise.at<float> (i, j) = noise.at<float> (i, j) +  (a * a + b * b) / 2;
	    }

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    noise.at<float> (i, j) = sqrt(noise.at<float> (i, j) / L);
	    if (check)
	    {
		if (input.type() == CV_8U)
		{
		    if (min > input.at<unsigned char>(i, j))
			min = input.at<unsigned char>(i, j);
		    if (max < input.at<unsigned char>(i, j))
			max = input.at<unsigned char>(i, j);
		}
		else
		{
		if (min > input.at<unsigned short>(i, j))
		    min = input.at<unsigned short>(i, j);
		}
	    }
	    float value;
	    if (input.type() == CV_8U)
		value = (float)input.at<unsigned char>(i, j) * noise.at<float> (i, j);
	    else
		value = (float)input.at<unsigned short>(i, j) * noise.at<float> (i, j);
	    if (input.type() == CV_8U)
	    {
		unsigned char truncated;
		if (value < 0)
		    truncated = 0;
		else
		    if (value > 255)
			truncated = 255;
		    else
			truncated = value;
		output.at<unsigned char>(i, j) = truncated;
	    }
	    else
	    {
		unsigned short truncated;
		if (value < 0)
		    truncated = 0;
		else
		    truncated = value;
		output.at<unsigned short>(i, j) = truncated;
	    }
	}

    if (check)
	checkDistrib(noise, min, max, false);

    return output;
}

cv::Mat
parseArgs (int argc, char* argv[], bool& check, bool& radar, int& outputArgc, float& param)
{
    //Wrong number of arguments.
    if (argc < 5)
    {
	printHelp();
	exit (1);
    }

    //Check radar mode
    if (argc > 2 && !strcmp(argv[2], "-radar"))
	radar = true;
    else
	radar = false;

    cv::Mat input;
    if (radar)
    {
	param = atof(argv[3]);
	input = ReadImw(argv[4]);
	outputArgc = 5;
    }
    else
    {
	param = atof(argv[2]);
	input = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	outputArgc= 4;
    }

    if (argc > outputArgc + 1 && !strcmp(argv[outputArgc + 1], "-check"))
	check = true;

    return (input);
}

int main (int argc, char* argv[])
{
    if (argc <= 1)
    {
	printHelp();
	return (1);
    }

    else if (!strcmp(argv[1], "gaussian"))
    { //Add gaussian noise
	float stddev;
	bool check;
	bool radar;
	int outputArgc;

	cv::Mat input = parseArgs(argc, argv, check, radar, outputArgc, stddev);

	cv::Mat output = gaussian(stddev, input, check);
	if (radar)
	    WriteImw(output, argv[outputArgc]);
	else
	    cv::imwrite(argv[outputArgc], output);
    }

    else if (!strcmp(argv[1], "rayleigh"))
    { //Add rayleigh noise
	float stddev;
	bool check;
	bool radar;
	int outputArgc;

	cv::Mat input = parseArgs(argc, argv, check, radar, outputArgc, stddev);

	cv::Mat output = rayleigh(stddev, input, check);
	if (radar)
	    WriteImw(output, argv[outputArgc]);
	else
	    cv::imwrite(argv[outputArgc], output);
    }

    else if (!strcmp(argv[1], "nakagami"))
    { //Add nakagami noise
	float L;
	bool check;
	bool radar;
	int outputArgc;

	cv::Mat input = parseArgs(argc, argv, check, radar, outputArgc, L);

	cv::Mat output = nakagami((int)L, input, check);
	if (radar)
	    WriteImw(output, argv[outputArgc]);
	else
	    cv::imwrite(argv[outputArgc], output);
    }

    else
    {
	printHelp();
	return (1);
    }

    return (0);
}
