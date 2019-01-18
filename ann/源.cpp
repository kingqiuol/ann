#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "ann.h"

using namespace std;
using namespace cv;

int main()
{
	NN ann(2);
	//ann.data_loader("./chinese_parts_train.txt");
	//ann.train("./bezdekIris.txt", {10,10,10,3});
	//ann.train("./chinese_parts_train.txt", {100,50});

	RNG rnger(getTickCount());
	Mat W(6, 3, CV_32FC1);
	rnger.fill(W, RNG::UNIFORM, cv::Scalar::all(0.),
		cv::Scalar::all(1.));
	cout << W << endl;

	//Mat row;
	//reduce(W, row, 1, REDUCE_MAX, CV_32FC1);
	//
	//Mat predict = ann.argmaxes(W);
	//cout << predict << endl;

	//Mat mask = Mat::zeros(3, 1, CV_8UC1);
	//cout << mask << endl;
	//Mat r;
	//float sum = 0.0f;
	//for (int i = 0; i < predict.rows; ++i){
	//	if (predict.at<uchar>(i, 0) == mask.at<uchar>(i, 0))
	//	{
	//		++sum;
	//	}
	//}
	//cout << sum << endl;

	//srand(time(0));
	//Mat m;
	//vector<Mat> mat;
	//for (int i = 0; i < 3; ++i){
	//	int j = rand() % 6;
	//	mat.push_back(W.row(j).clone());
	//}
	//vconcat(mat, m);
	//cout << m << endl;
	
	return 0;
}

