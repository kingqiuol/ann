#include "ann.h"

void NN::data_loader(const string &path)
{
	ifstream file(path);
	//将数据存储到vector中
	vector<vector<float>> dataset;
	string ss;
	while (getline(file, ss)){
		vector<float> data;

		stringstream w(ss);
		float temp;
		while (w >> temp){
			data.push_back(temp);
		}
		if (data.size() == 0){
			continue;
		}
		dataset.push_back(data);
	}

	//随机打乱数据
	srand(static_cast<unsigned int>(time(0)));
	random_shuffle(dataset.begin(), dataset.end());
	
	//将vector转化为Mat并分别存储到训练集和label中
	int rows = static_cast<int>(dataset.size());
	int cols = static_cast<int>(dataset[0].size() - 1);

	//创建数据集和label矩阵
	Mat train_data(rows, cols, CV_32FC1);
	Mat labels(rows, 1, CV_32FC1);
	//加载数据
	auto it = dataset.begin();
	for (int i = 0; i < rows; ++i){
		float *data = train_data.ptr<float>(i);
		float *label = labels.ptr<float>(i);
		for (int j = 0; j < cols + 1; ++j){
			data[j] = (*it)[j];
			if (cols == j){
				label[0] = (*it)[j];
			}
		}
		++it;
	}

	//将共享指针指向数据
	this->data_ptr = make_shared<Mat>(train_data);
	this->label_ptr = make_shared<Mat>(labels);
}

void NN::add_hidden_layer(const vector<size_t> &num_hiddens)
{
	hiddens_.clear();
	hiddens_.assign(num_hiddens.begin(), num_hiddens.end());
}

void NN::initial_layer(const size_t &input, const size_t &output)
{
	RNG rnger(getTickCount());
	//创建权重矩阵
	Mat W(static_cast<int>(input), static_cast<int>(output), CV_32FC1);
	//创建偏置矩阵
	Mat b(1,static_cast<int>(output), CV_32FC1, 1.f);
	//权重的初始化
	rnger.fill(W, RNG::UNIFORM, cv::Scalar::all(-1.f / sqrt(input)), cv::Scalar::all(1.f / sqrt(input)));

	W_.push_back(W);
	b_.push_back(b);
}

void NN::initial_networks()
{
	//初始化输入神经元个数
	//如果未设置初始神经元个数
	this->input_ = this->data_ptr->cols;

	//初始化权重
	RNG rnger(getTickCount());
	if (this->hiddens_.size()){
		size_t cur_row = this->input_;
		auto it = this->hiddens_.begin();
		while (it != this->hiddens_.end()){
			initial_layer(cur_row, *it);
			cur_row = *it;
			++it;
		}
		initial_layer(cur_row, this->classes_);

	}
	else{
		initial_layer(this->input_, this->classes_);
	}
}

void NN::relu(Mat &X)
{
	//生成大于零的模板并归一化
	Mat mask = (X > 0) / 255;
	//将矩阵转化为float类型
	Mat mask2f;
	mask.convertTo(mask2f, CV_32FC1);
	//relu计算输出
	multiply(X, mask2f, X);
}

Mat NN::mat_sum(const Mat &X, const int &axis, const int &dtype)
{
	Mat sum, sum_expend;
	reduce(X, sum, axis, REDUCE_SUM, dtype);
	if (axis){
		repeat(sum, 1, X.cols, sum_expend);
	}
	else{
		repeat(sum, X.rows, 1, sum_expend);
	}

	return sum_expend;
}

Mat NN::mat_max(const Mat &X, const int &axis, const int &dtype)
{
	Mat max, max_expend;
	//获取每一行的最大值
	reduce(X, max, axis, REDUCE_MAX, dtype);
	//沿x方向进行维度扩张并拷贝数据
	if (axis){
		repeat(max, 1, X.cols, max_expend);
	}
	else{
		repeat(max, X.rows, 1, max_expend);
	}
	return max_expend;
}

int NN::argmax(Mat &row, float &max_value)
{
	int max_point2f[2] = { 0 };
	double dmax = static_cast<double>(max_value);
	minMaxIdx(row, 0, &dmax, 0, max_point2f);

	return max_point2f[1];
}

Mat NN::argmaxes(Mat &out)
{
	Mat max_row;
	reduce(out, max_row, 1, REDUCE_MAX, CV_32FC1);

	Mat predict(out.rows, 1, CV_32FC1);
	for (int i = 0; i < out.rows; i++){
		float *data = predict.ptr<float>(i);
		float *max = max_row.ptr<float>(i);
		data[0] = static_cast<float>(argmax(out.row(i), max[0]));
	}

	return predict;
}

void NN::softmax(Mat &out)
{
	//输出减去最大值防止指数爆炸
	Mat row_max_expend = mat_max(out, 1, CV_32FC1);
	out -= row_max_expend;

	exp(out, out);

	//归一化
	Mat row_sum_expend = mat_sum(out, 1, CV_32FC1);
	out /= row_sum_expend;
}

void NN::forward(Mat &X)
{
	this->out_.clear();//清空当前存储的输出
	this->out_.push_back(X);
	auto it_w = this->W_.begin();
	auto it_b = this->b_.begin();

	Mat out = X;
	size_t num_layers = 1;
	for (; it_w != this->W_.end(); ++it_w, ++it_b){
		out = out*(*it_w);
		Mat b;
		repeat(*it_b, out.rows, 1, b);
		out += b;

		if (num_layers < this->W_.size()){
			this->relu(out);
			this->out_.push_back(out);//存储中间输出结果
		}

		++num_layers;
	}

	//计算softmax输出
	softmax(out);
	this->out_.push_back(out);
}

float NN::L2_regular()
{
	auto it = this->W_.begin();
	float regular = 0.0;
	while (it != this->W_.end()){
		Mat temp;
		multiply(*it, *it, temp);
		regular += static_cast<float>(sum(temp)[0]);
		++it;
	}

	return regular*this->reg_;
}

float NN::loss(Mat &y)
{
	float sum = 0.0;
	Mat out = this->out_.back();
	for (int i = 0; i < y.rows; ++i){
		float* data = y.ptr<float>(i);
		float* predict = out.ptr<float>(i);

		sum -= log(predict[static_cast<int>(data[0])]);
	}
	sum /= out.rows;

	return sum + L2_regular();
}

void NN::backward()
{
	auto it_out = out_.rbegin();		//获取存储输出容器的迭代器
	auto it_w = W_.rbegin();			//获取存储权重容器的迭代器
	int num_trains = it_out->rows;		//训练集的总数
	dW_.clear(); db_.clear();			//清空当前存储梯度的容器

	Mat dL = (*it_out).clone();//后层的累积梯度
	//计算softmax分类层的梯度
	for (int i = 0; i < dL.rows; ++i){
		float *data = dL.ptr<float>(i);
		float *l = label_ptr->ptr<float>(i);

		data[static_cast<int>(l[0])] -= 1;
	}

	Mat db;			//偏置项的梯度
	Mat dw;			//权重的梯度
	Mat W = *it_w;	//保存上一层的权重
	while (it_w != W_.rend()){
		++it_out;

		if (it_w != W_.rbegin()){
			dL = dL*W.t();
			Mat d_relu = (*(it_out-1)) > 0 / 255;//计算当前层relu处的梯度
			d_relu.convertTo(d_relu, CV_32FC1);
			multiply(d_relu, dL, dL);
			W = *it_w;
		}

		//计算当前层偏置项梯度
		reduce(dL, db, 0, REDUCE_SUM, CV_32FC1);
		db /= num_trains;

		//计算当前层权重的梯度
		dw = (*it_out).t()*dL / num_trains;
		dw += 2 * reg_*(*it_w);

		//更新各层的偏置项梯度和权重的梯度
		db_.insert(db_.begin(), db);
		dW_.insert(dW_.begin(), dw);

		++it_w;
	}
}

void NN::update_weight()
{
	//逐层更新
	auto it_w = W_.begin();
	auto it_b = b_.begin();
	auto it_dw = dW_.begin();
	auto it_db = db_.begin();

	while (it_w != W_.end()){
		*it_w -= learning_rate_*(*it_dw);
		*it_b -= learning_rate_*(*it_db);

		++it_w;
		++it_b;
		++it_dw;
		++it_db;
	}
}

void NN::get_batch(Mat &batch_X, Mat &batch_y)
{
	srand((unsigned int)time(0));
	int index = 0;
	vector<Mat> vec_x,vec_y;
	for (int i = 0; i < batch_size_; ++i){
		index = rand() % data_ptr->rows;
		vec_x.push_back(data_ptr->row(index).clone());
		vec_y.push_back(label_ptr->row(index).clone());
	}

	vconcat(vec_x, batch_X);
	vconcat(vec_y, batch_y);
}

void NN::save_weights(const string &save_path)
{
	//创建xml
	FileStorage fs(save_path, FileStorage::WRITE);
	int size = static_cast<int>(W_.size());
	fs << "size" << size;

	for (int i = 0; i < size; ++i){
		string weights = "W" + to_string(i);
		string bias = "b" + to_string(i);
		fs << weights << W_[i];
		fs << bias << b_[i];
	}

	fs.release();
}

void NN::load_weights(const string &load_path)
{
	FileStorage fr(load_path, FileStorage::READ);
	int size;
	fr["size"] >> size;
	
	for (int i = 0; i < size; ++i){
		Mat W,b;
		string weights = "W" + to_string(i);
		string bias = "b" + to_string(i);

		fr[weights] >> W;
		fr[bias] >> b;
		W_.push_back(W);
		b_.push_back(b);
	}

	fr.release();
}

void NN::train(const string &file_path, const vector<size_t> &num_hiddens)
{
	//加载数据
	data_loader(file_path);

	//创建网络
	add_hidden_layer(num_hiddens);

	//初始化网络
	initial_networks();

	size_t epoch = 0;
	float train_loss = 0.0f;
	float max_accuracy = 0.0f;
	while (epoch < max_epochs_){	
		//Mat batch_X,batch_y;
		//get_batch(batch_X, batch_y);

		//forward(batch_X);
		forward(*data_ptr);
		backward();
		update_weight();

		//train_loss = loss(batch_y);
		train_loss = loss(*label_ptr);
		cout << "epoch:" << epoch << " ,loss:" << train_loss << endl;
		if ((epoch+1) % 10 == 0){
			forward(*data_ptr);
			Mat out = out_.back();
			Mat predict = argmaxes(out);
			float sum = 0.0f;
			for (int i = 0; i < predict.rows; ++i){
				if (predict.at<float>(i, 0) == label_ptr->at<float>(i, 0)){
					++sum;
				}
			}
			
			float accuracy = (sum / predict.rows) * 100;
			cout << "accuracy:" << (sum / predict.rows) * 100 << "%" << endl;
			if (max_accuracy < accuracy){
				save_weights("./ann_model.xml");
			}
		}

		++epoch;
	}
}

Mat NN::predict(Mat &data)
{
	forward(data);
	
	return out_.back();
}

