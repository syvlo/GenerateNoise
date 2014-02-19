#include <iostream>
#include <chrono>
#include <random>

#include <cstring>
#include <cstdlib>

#include <cv.h>
#include <highgui.h>


void
printHelp()
{
    std::cout << "Generate noise on images." << std::endl
	      << "Usage:" << std::endl
	      << "./GenerateNoise gaussian stddev input output" << std::endl
	      << "./GenerateNoise rayleigh stddev input output" << std::endl;
}

void
checkDistrib(const cv::Mat noise)
{
    unsigned width = noise.size().width;
    unsigned height = noise.size().height;
    double nbPix = width * height;

    double mean = 0;

    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    mean += noise.at<double>(i, j);
	}
    mean /= nbPix;

    std::cout << "Mean = " << mean << std::endl;

    double std = 0;
    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    std += (noise.at<double>(i, j) - mean) * (noise.at<double>(i, j) - mean);
	}
    std /= nbPix;
    std = sqrt(std);

    std::cout << "Standart deviation = " << std << std::endl;
    std::cout << "Coefficient of variation = " << std / mean << std::endl;
}

cv::Mat
gaussian(double stddev, const cv::Mat input, bool check)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);

    std::normal_distribution<double> distribution (0.0,stddev);

    unsigned width = input.size().width;
    unsigned height = input.size().height;

    cv::Mat output(input.size(), CV_8U);
    cv::Mat noise(input.size(), CV_32F);
    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    double noiseValue = distribution(generator);
	    if (check)
		noise.at<double>(i, j) = noiseValue;
	    double value = input.at<unsigned char>(i, j) + noiseValue;
	    output.at<unsigned char>(i, j) = ((value > 255 ? 255 : value) < 0 ? 0 : value);
	}

    if (check)
	checkDistrib(noise);

    return output;
}

cv::Mat
rayleigh(double sigma, const cv::Mat input, bool check)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    unsigned width = input.size().width;
    unsigned height = input.size().height;

    cv::Mat output(input.size(), CV_8U);
    cv::Mat noise(input.size(), CV_32F);
    for (unsigned i = 0; i < height; ++i)
	for (unsigned j = 0; j < width; ++j)
	{
	    double noiseValue = sigma * sqrt(-2 * log(distribution(generator)));
	    if (check)
		noise.at<double>(i, j) = noiseValue;
	    double value = (double)input.at<unsigned char>(i, j) * noiseValue;
	    output.at<unsigned char>(i, j) = ((value > 255 ? 255 : value) < 0 ? 0 : value);
	}

    if (check)
	checkDistrib(noise);

    return output;
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
	if (argc != 5)//Wrong number of arguments.
	{
	    printHelp();
	    return (1);
	}
	double stddev = atof(argv[2]);
	cv::Mat input = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat output = gaussian(stddev, input, true);
	cv::imwrite(argv[4], output);
	return (0);
    }

    else if (!strcmp(argv[1], "rayleigh"))
    { //Add gaussian noise
	if (argc != 5)//Wrong number of arguments.
	{
	    printHelp();
	    return (1);
	}
	double stddev = atof(argv[2]);
	cv::Mat input = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat output = rayleigh(stddev, input, true);
	cv::imwrite(argv[4], output);
	return (0);
    }

    else
    {
	printHelp();
	return (1);
    }
}
