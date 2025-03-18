#include <iomanip>
#include <direct.h>
#include "NAM.h"
#include "convertToMat.h"
#include "getImageList.h"
#include <iostream>



int main()
{
	string inDir;
	string outDir;
	cout << "Please input the path of the input folder: ";
	cin >> inDir;
	cout << "Please input the path of the output folder: ";
	cin >> outDir;

	inDir = inDir + '/';
	outDir = outDir + '/';

	/*_mkdir(outDir.c_str());*/

	unordered_map<string, double> param;

	/* Output */
	param["mode"] = 1;		// 0: Only label 1: Output images and label
	param["saveHir"] = 60;	// Number of saved images (in our paper it is 60)

	/* Rgb2lab */
	param["lab"] = 0;					// Light source 0:D50 1:D65

	/* Representation */
	param["homogeneous"] = 0;			// Dissimilarity between two pixels. 0:Euclidean Eq.(10) 1:Gouraud Eq. (11)
	param["k"] = 2.0;					// Threshold in Eq. (10) or Eq. (11)
	param["shape"] = 0;					// Shape of homogeneous blocks 0:Square 1:Diagonal 2:Vertical 3:Diagonal
	param["ratio"] = 1.0;				// Maximum ratio of height to width of homogeneous blocks

	/* Mergence */
	param["u"] = 1.9;					// Mean
	param["var"] = 3.0;					// Variance

	/* Scanning */
	param["alpha"] = 1.0;				//
	param["beta"] = 1.97;				//
	param["gamma"] = 1.97;				//
	param["percent"] = 6.7;				// Scanning
	param["si"] = 67.0;					//
	param["size"] = 4700.0;				//
	param["change"] = 197;				//

	/* Display in terminal */
	param["seg"] = 0;					// Number of segments after removing remnant regions
	param["remnant"] = 0;
	param["scanning"] = 0;				// Cost time
	param["representation"] = 0;
	param["mergence"] = 0;

	double avgPhase1Time, avgPhase2Time, avgCostTime, avgSeg;
	avgPhase1Time = avgPhase2Time = avgCostTime = avgSeg = 0;
	int count = 1;

	vector<string> imageList;
	getImageList(inDir.c_str(), imageList);

	for (auto& x : imageList) {
		param["seg"] = 0;
		param["representation"] = 0;
		param["mergence"] = 0;
		param["remnant"] = 0;
		param["scanning"] = 0;

		vector<Mat> map;
		vector<Mat> result;

		NAM(param, inDir + x, map, result);

		if (param["mode"] == 0)
		{
			convertToMat(map, outDir + x.substr(0, x.size() - 4) + ".mat");
			for (auto& x : map)
				x.release();
		}
		else if (param["mode"] == 1)
		{
			Mat img = imread(inDir + x);
			if (!img.data || img.channels() != 3)
				return -1;
			const int& M = img.rows;
			const int& N = img.cols;

			{
				Mat mark = Mat::zeros(M, N, CV_8UC1);
				for (int m = 0; m < M; m++)
					for (int n = 0; n < N; n++)
						if (!mark.at<uchar>(m, n) && m >= 1 && m < M - 1 && map[0].at<float>(m - 1, n) != map[0].at<float>(m + 1, n) && !mark.at<uchar>(m - 1, n) && !mark.at<uchar>(m + 1, n)
							|| n >= 1 && n < N - 1 && map[0].at<float>(m, n - 1) != map[0].at<float>(m, n + 1) && !mark.at<uchar>(m, n - 1) && !mark.at<uchar>(m, n + 1))
						{
							mark.at<uchar>(m, n) = 255;
						}

				imwrite(outDir + x.substr(0, x.size() - 4) + string("_Hier_") + to_string(result.size()) + string("_Mark") + ".png", mark);

				mark.release();
			}

			for (auto& x : map)
				x.release();
		}

		avgSeg += param["seg"];

		double p1, p2, cost;
		p1 = param["representation"] + param["mergence"] + param["remnant"];
		p2 = param["scanning"];
		cost = p1 + p2;

		std::cout << setw(4) << count++ << " ";
		std::cout << setw(10) << x.substr(0, x.size() - 4) << "\t";
		std::cout << setw(4) << cost << " s" << endl;

		avgPhase1Time += p1;
		avgPhase2Time += p2;
		avgCostTime += cost;
	}
	count--;
	std::cout << "Average numSeg :" << avgSeg / (1.0 * count) << endl;
	std::cout << "Average phase1Time: " << avgPhase1Time / (1.0 * count) << " s" << endl;
	std::cout << "Average phase2Time: " << avgPhase2Time / (1.0 * count) << " s" << endl;
	std::cout << "Average costTime: " << avgCostTime / (1.0 * count) << " s" << endl;

	return 0;
}


//BOOST_PYTHON_MODULE(NAMLab)
//{
//	using namespace boost::python;
//    def("namLab", namLab);
//}