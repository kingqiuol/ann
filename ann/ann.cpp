#include "ann.h"

void NN::data_loader(const string &path)
{
	ifstream file(path);
	//�����ݴ洢��vector��
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

	//�����������
	srand(static_cast<unsigned int>(time(0)));
	random_shuffle(dataset.begin(), dataset.end());
	
	//��vectorת��ΪMat���ֱ�洢��ѵ������label��
	int rows = static_cast<int>(dataset.size());
	int cols = static_cast<int>(dataset[0].size() - 1);

	//�������ݼ���label����
	Mat train_data(rows, cols, CV_32FC1);
	Mat labels(rows, 1, CV_32FC1);
	//��������
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

	//������ָ��ָ������
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
	//����Ȩ�ؾ���
	Mat W(static_cast<int>(input), static_cast<int>(output), CV_32FC1);
	//����ƫ�þ���
	Mat b(1,static_cast<int>(output), CV_32FC1, 1.f);
	//Ȩ�صĳ�ʼ��
	rnger.fill(W, RNG::UNIFORM, cv::Scalar::all(-1.f / sqrt(input)), cv::Scalar::all(1.f / sqrt(input)));

	W_.push_back(W);
	b_.push_back(b);
}

void NN::initial_networks()
{
	//��ʼ��������Ԫ����
	//���δ���ó�ʼ��Ԫ����
	this->input_ = this->data_ptr->cols;

	//��ʼ��Ȩ��
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
	//���ɴ������ģ�岢��һ��
	Mat mask = (X > 0) / 255;
	//������ת��Ϊfloat����
	Mat mask2f;
	mask.convertTo(mask2f, CV_32FC1);
	//relu�������
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
	//��ȡÿһ�е����ֵ
	reduce(X, max, axis, REDUCE_MAX, dtype);
	//��x�������ά�����Ų���������
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
	//�����ȥ���ֵ��ָֹ����ը
	Mat row_max_expend = mat_max(out, 1, CV_32FC1);
	out -= row_max_expend;

	exp(out, out);

	//��һ��
	Mat row_sum_expend = mat_sum(out, 1, CV_32FC1);
	out /= row_sum_expend;
}

void NN::forward(Mat &X)
{
	this->out_.clear();//��յ�ǰ�洢�����
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
			this->out_.push_back(out);//�洢�м�������
		}

		++num_layers;
	}

	//����softmax���
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
	auto it_out = out_.rbegin();		//��ȡ�洢��������ĵ�����
	auto it_w = W_.rbegin();			//��ȡ�洢Ȩ�������ĵ�����
	int num_trains = it_out->rows;		//ѵ����������
	dW_.clear(); db_.clear();			//��յ�ǰ�洢�ݶȵ�����

	Mat dL = (*it_out).clone();//�����ۻ��ݶ�
	//����softmax�������ݶ�
	for (int i = 0; i < dL.rows; ++i){
		float *data = dL.ptr<float>(i);
		float *l = label_ptr->ptr<float>(i);

		data[static_cast<int>(l[0])] -= 1;
	}

	Mat db;			//ƫ������ݶ�
	Mat dw;			//Ȩ�ص��ݶ�
	Mat W = *it_w;	//������һ���Ȩ��
	while (it_w != W_.rend()){
		++it_out;

		if (it_w != W_.rbegin()){
			dL = dL*W.t();
			Mat d_relu = (*(it_out-1)) > 0 / 255;//���㵱ǰ��relu�����ݶ�
			d_relu.convertTo(d_relu, CV_32FC1);
			multiply(d_relu, dL, dL);
			W = *it_w;
		}

		//���㵱ǰ��ƫ�����ݶ�
		reduce(dL, db, 0, REDUCE_SUM, CV_32FC1);
		db /= num_trains;

		//���㵱ǰ��Ȩ�ص��ݶ�
		dw = (*it_out).t()*dL / num_trains;
		dw += 2 * reg_*(*it_w);

		//���¸����ƫ�����ݶȺ�Ȩ�ص��ݶ�
		db_.insert(db_.begin(), db);
		dW_.insert(dW_.begin(), dw);

		++it_w;
	}
}

void NN::update_weight()
{
	//������
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
	//����xml
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
	//��������
	data_loader(file_path);

	//��������
	add_hidden_layer(num_hiddens);

	//��ʼ������
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

