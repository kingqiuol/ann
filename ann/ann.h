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

	//加载数据
	void data_loader(const string &path);
	//添加隐藏层
	void add_hidden_layer(const vector<size_t> &num_hiddens = {});
	//初始化网络
	void initial_networks();

	//前向传播
	void forward(Mat &X);
	//反向传播
	void backward();
	//计算损失函数
	float loss(Mat &y);
	//更新权重
	void update_weight();

	//训练网络
	void train(const string &file_path, const vector<size_t> &num_hiddens);

	//保存权重
	void save_weights();
	//加载权重
	void load_weights();

	//预测
	void predict();

	//其他方法
	inline float get_learning_rate()const{ return this->learning_rate_; }
	inline void set_learning_rate(float learning_rate){ this->learning_rate_ = learning_rate_; }
	inline float get_reg() const{ return this->reg_; }
	inline void set_reg(float reg){ this->reg_ = reg; }
	inline size_t get_epoch()const{ return this->max_epochs_; }
private:
	/***神经网络相关的函数***/
	void get_batch(Mat &batch_X,Mat &batch_y);
	void initial_layer(const size_t &input, const size_t &output);//单个层的初始化
	void relu(Mat &X);//激励函数
	void softmax(Mat &out);//softmax分类器
	float L2_regular();//L2正则化

	/***其他与矩阵操作相关的方法***/
	//计算矩阵行/列方向的和，并进行矩阵增广
	Mat mat_sum(const Mat &X, const int &axis, const int &dtype);
	//计算矩阵行/列方向的最大值，并进行矩阵增广
	Mat mat_max(const Mat &X, const int &axis, const int &dtype);
	//求矩阵行对对应的最大值的下标所在的列
	int argmax(Mat &row, float &max);//单行对应的下标
	Mat argmaxes(Mat &out);

	/***常见神经网络参数设置***/
	float reg_;					//正则化系数
	float learning_rate_;		//学习率
	size_t max_epochs_;			//最大训练次数
	size_t batch_size_;			//批量处理大小

private:
	//输入数据、数据标签
	shared_ptr<Mat> data_ptr, label_ptr;

	size_t input_;				//输入个数
	size_t classes_;			//分类个数
	vector<size_t> hiddens_;	//各个隐藏层中神经元个数

	vector<Mat> W_;				//保存权重
	vector<Mat> b_;				//保存偏置项
	vector<Mat> out_;			//存储各个层的输出
	vector<Mat> dW_;			//保存各个层的计算权重梯度
	vector<Mat> db_;			//保存各个层的偏置项梯度
};


#endif // !ANN_H_
