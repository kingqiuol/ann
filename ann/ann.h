#ifndef ANN_H_
#define ANN_H_

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class NN{
public:
	NN(size_t classes, size_t input = 0, 
		float reg = 0.005, float learning_rate = 0.0001,
		size_t  max_epochs = 5000,size_t batch_size = 500) :
		input_(input), classes_(classes),
		reg_(reg), learning_rate_(learning_rate),
		max_epochs_(max_epochs), batch_size_(batch_size),
		data_ptr(nullptr), label_ptr(nullptr){}
	~NN(){}

	//��������
	void data_loader(const string &path);
	//������ز�
	void add_hidden_layer(const vector<size_t> &num_hiddens = {});
	//��ʼ������
	void initial_networks();

	//ǰ�򴫲�
	void forward(Mat &X);
	//���򴫲�
	void backward();
	//������ʧ����
	float loss(Mat &y);
	//����Ȩ��
	void update_weight();

	//ѵ������
	void train(const string &file_path, const vector<size_t> &num_hiddens);

	//����Ȩ��
	void save_weights();
	//����Ȩ��
	void load_weights();

	//Ԥ��
	void predict();

	//��������
	inline float get_learning_rate()const{ return this->learning_rate_; }
	inline void set_learning_rate(float learning_rate){ this->learning_rate_ = learning_rate_; }
	inline float get_reg() const{ return this->reg_; }
	inline void set_reg(float reg){ this->reg_ = reg; }
	inline size_t get_epoch()const{ return this->max_epochs_; }
private:
	/***��������صĺ���***/
	void get_batch(Mat &batch_X,Mat &batch_y);
	void initial_layer(const size_t &input, const size_t &output);//������ĳ�ʼ��
	void relu(Mat &X);//��������
	void softmax(Mat &out);//softmax������
	float L2_regular();//L2����

	/***��������������صķ���***/
	//���������/�з���ĺͣ������о�������
	Mat mat_sum(const Mat &X, const int &axis, const int &dtype);
	//���������/�з�������ֵ�������о�������
	Mat mat_max(const Mat &X, const int &axis, const int &dtype);
	//������жԶ�Ӧ�����ֵ���±����ڵ���
	int argmax(Mat &row, float &max);//���ж�Ӧ���±�
	Mat argmaxes(Mat &out);

	/***�����������������***/
	float reg_;					//����ϵ��
	float learning_rate_;		//ѧϰ��
	size_t max_epochs_;			//���ѵ������
	size_t batch_size_;			//���������С

private:
	//�������ݡ����ݱ�ǩ
	shared_ptr<Mat> data_ptr, label_ptr;

	size_t input_;				//�������
	size_t classes_;			//�������
	vector<size_t> hiddens_;	//�������ز�����Ԫ����

	vector<Mat> W_;				//����Ȩ��
	vector<Mat> b_;				//����ƫ����
	vector<Mat> out_;			//�洢����������
	vector<Mat> dW_;			//���������ļ���Ȩ���ݶ�
	vector<Mat> db_;			//����������ƫ�����ݶ�
};


#endif // !ANN_H_
