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

	ann.load_weights("./ann_model.xml");

	vector<double> v = { 0.2797297297297297, 0.925, 0.1984732824427481,
		-2.1521224327220794, -1.1124497599094805, -0.07277708709688184,
		-0.4886461562219213, -0.4886461562219213, 0.34309198202815766,
		0.7589610511531971, 0.34309198202815766, -0.07277708709688184,
		-0.6965806907844411, -1.3203842944720003, -0.9045152253469608,
		-0.28071162165940156, 0.7589610511531971, 2.214502793090835,
		2.214502793090835, 1.3827646548407564, 1.798633723965796,
		1.3827646548407564, -0.4886461562219213, -0.28071162165940156,
		-0.07277708709688184, -0.28071162165940156, -0.07277708709688184,
		-0.28071162165940156, -0.07277708709688184, -0.07277708709688184,
		0.1351574474656379, 2.630371862215875, -0.07277708709688184,
		-0.07277708709688184, 0.1351574474656379, 0.34309198202815766,
		-0.07277708709688184, -0.9045152253469608, -0.9045152253469608,
		-1.1124497599094805, -1.1124497599094805, -0.6965806907844411,
		-0.28071162165940156, 0.9113065579313208, -0.9980976586866847,
		1.0848887594420487, -0.9980976586866847, 1.5460186011553896,
		-0.8834392006602226, 0.22085980016505566, -0.8834392006602226,
		1.5075339959587122, -0.9045203975752274, 0.30150679919174245,
		-0.9045203975752274, 0.6847489863952442, -0.9782128377074917,
		1.2716766890197393, -0.9782128377074917 };	
	Mat data=Mat(v).t();
	data.convertTo(data, CV_32FC1);
	Mat pred=ann.predict(data);
	cout << pred << endl;


	//RNG rnger(getTickCount());
	//Mat W(3, 3, CV_32FC1);
	//rnger.fill(W, RNG::UNIFORM, cv::Scalar::all(0.),
	//	cv::Scalar::all(1.));
	//cout << W << endl;

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
	
	//FileStorage fs("./1.xml", FileStorage::WRITE); 
	//int size = 3;

	//for (int i = 0; i < size; ++i){
	//	string w = "W" + to_string(i);
	//	string b = "b" + to_string(i);
	//	fs << w << W;
	//}

	//fs.release();
	//
	//FileStorage fr("./1.xml", FileStorage::READ);
	//Mat mat1,mat2;
	//
	//for (int j = 0; j < size; ++j){
	//	string w = "W" + to_string(j);
	//	cout << w << endl;
	//	fr[w] >> mat1;
	//	cout << mat1<<endl;
	//}

	//fr.release();
	return 0;
}

