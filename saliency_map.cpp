#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>

using namespace std;
using namespace cv;

const string path = "C:\\bin_OpenCV\\bin\\workspace\\opencv_test\\test_proj\\Debug\\lena.jpg"; //���͉摜�̃p�X

const int STEP = 8;
const int GABOR_R = 8; //�K�{�[���J�[�l���̃T�C�Y�i���a�j
const float WEIGHT_I = 0.333f; //�P�x�}�b�v�̏d�݌W��
const float WEIGHT_C = 0.333f; //�F���}�b�v�̏d�݌W��
const float WEIGHT_O = 0.333f; //�����}�b�v�̏d�݌W��

//�X�P�[���̈قȂ�Q�̉摜�ɂ��� "center-surround" ���Z
Mat operateCenterSurround(const Mat& center, const Mat& surround)
{
	Mat csmap(center.size(), center.type());
	resize(surround, csmap, csmap.size()); //surround�摜��center�摜�Ɠ��T�C�Y�Ɋg��
	csmap = abs(csmap - center);
	return csmap;
}

//�e��s���~�b�h���� "center-surround" �s���~�b�h���\�z
vector<Mat> buildCenterSurroundPyramid(const vector<Mat>& pyramid)
{
	//surround=center+delta, center={2,3,4}, delta={3,4} �v6��
	vector<Mat> cspyr(6);
	cspyr[0] = operateCenterSurround(pyramid[2], pyramid[5]);
	cspyr[1] = operateCenterSurround(pyramid[2], pyramid[6]);
	cspyr[2] = operateCenterSurround(pyramid[3], pyramid[6]);
	cspyr[3] = operateCenterSurround(pyramid[3], pyramid[7]);
	cspyr[4] = operateCenterSurround(pyramid[4], pyramid[7]);
	cspyr[5] = operateCenterSurround(pyramid[4], pyramid[8]);
	return cspyr;
}

//�摜�̃_�C�i�~�b�N�����W��[0,1]�ɐ��K��
void normalizeRange(Mat& image)
{
	double minval, maxval;
	minMaxLoc(image, &minval, &maxval);

	image -= minval;
	if (minval<maxval)
		image /= maxval - minval;
}

//���K�����Z�qN(�E)�F�V���O���s�[�N�̋����ƃ}���`�s�[�N�̗}��
void trimPeaks(Mat& image, int step)
{
	const int w = image.cols;
	const int h = image.rows;

	const double M = 1.0;
	normalizeRange(image);
	double m = 0.0;
	for (int y = 0; y<h - step; y += step) //�[��(h%step)�����]��
		for (int x = 0; x<w - step; x += step) //�[��(w%step)�����]��
		{
			Mat roi(image, Rect(x, y, step, step));
			double minval = 0.0, maxval = 0.0;
			minMaxLoc(roi, &minval, &maxval);
			m += maxval;
		}
	m /= (w / step - (w%step ? 0 : 1))*(h / step - (h%step ? 0 : 1)); //�u���b�N���Ŋ����ĕ��ς��v�Z
	image *= (M - m)*(M - m);
}

//�������}�b�v���v�Z����
Mat calcSaliencyMap(const Mat& image0)
{
	const Mat_<Vec3f> image = image0 / 255.0f; //�_�C�i�~�b�N�����W�̐��K��

	//�K�{�[���J�[�l���̎��O����
	const Size ksize = Size(GABOR_R + 1 + GABOR_R, GABOR_R + 1 + GABOR_R);
	const double sigma = GABOR_R / CV_PI; //�}�΃Ђ܂ŃT�|�[�g����悤�ɒ���
	const double lambda = GABOR_R + 1; //�Б���1�����ɒ���
	const double deg45 = CV_PI / 4.0;
	Mat gabor000 = getGaborKernel(ksize, sigma, deg45 * 0, lambda, 1.0, 0.0, CV_32F);
	Mat gabor045 = getGaborKernel(ksize, sigma, deg45 * 1, lambda, 1.0, 0.0, CV_32F);
	Mat gabor090 = getGaborKernel(ksize, sigma, deg45 * 2, lambda, 1.0, 0.0, CV_32F);
	Mat gabor135 = getGaborKernel(ksize, sigma, deg45 * 3, lambda, 1.0, 0.0, CV_32F);

	const int NUM_SCALES = 9;
	vector<Mat> pyramidI(NUM_SCALES); //�P�x�s���~�b�h
	vector<Mat> pyramidRG(NUM_SCALES); //�F��RG�s���~�b�h
	vector<Mat> pyramidBY(NUM_SCALES); //�F��BY�s���~�b�h
	vector<Mat> pyramid000(NUM_SCALES); //����  0���s���~�b�h
	vector<Mat> pyramid045(NUM_SCALES); //���� 45���s���~�b�h
	vector<Mat> pyramid090(NUM_SCALES); //���� 90���s���~�b�h
	vector<Mat> pyramid135(NUM_SCALES); //����135���s���~�b�h

	//�����}�b�v�s���~�b�h�̍\�z
	Mat scaled = image; //�ŏ��̃X�P�[���͌��摜��
	for (int s = 0; s<NUM_SCALES; ++s)
	{
		const int w = scaled.cols;
		const int h = scaled.rows;

		//�P�x�}�b�v�̐���
		vector<Mat_<float> > colors;
		split(scaled, colors);
		Mat_<float> imageI = (colors[0] + colors[1] + colors[2]) / 3.0f;
		pyramidI[s] = imageI;

		//���K��rgb�l�̌v�Z
		double minval, maxval;
		minMaxLoc(imageI, &minval, &maxval);
		Mat_<float> r(h, w, 0.0f);
		Mat_<float> g(h, w, 0.0f);
		Mat_<float> b(h, w, 0.0f);
		for (int j = 0; j<h; ++j)
			for (int i = 0; i<w; ++i)
			{
				if (imageI(j, i)<0.1f*maxval) //�ő�s�[�N��1/10�ȉ��̉�f�͏��O
					continue;
				r(j, i) = colors[2](j, i) / imageI(j, i);
				g(j, i) = colors[1](j, i) / imageI(j, i);
				b(j, i) = colors[0](j, i) / imageI(j, i);
			}

		//�F���}�b�v�̐����i���l��0�ɃN�����v�j
		Mat R = max(0.0f, r - (g + b) / 2);
		Mat G = max(0.0f, g - (b + r) / 2);
		Mat B = max(0.0f, b - (r + g) / 2);
		Mat Y = max(0.0f, (r + g) / 2 - abs(r - g) / 2 - b);
		pyramidRG[s] = R - G;
		pyramidBY[s] = B - Y;

		//�����}�b�v�̐���
		filter2D(imageI, pyramid000[s], -1, gabor000);
		filter2D(imageI, pyramid045[s], -1, gabor045);
		filter2D(imageI, pyramid090[s], -1, gabor090);
		filter2D(imageI, pyramid135[s], -1, gabor135);

		pyrDown(scaled, scaled); //���̃I�N�^�[�u�Ɍ����ăX�P�[���_�E��
	}

	//center-surround���Z
	vector<Mat> cspyrI = buildCenterSurroundPyramid(pyramidI);
	vector<Mat> cspyrRG = buildCenterSurroundPyramid(pyramidRG);
	vector<Mat> cspyrBY = buildCenterSurroundPyramid(pyramidBY);
	vector<Mat> cspyr000 = buildCenterSurroundPyramid(pyramid000);
	vector<Mat> cspyr045 = buildCenterSurroundPyramid(pyramid045);
	vector<Mat> cspyr090 = buildCenterSurroundPyramid(pyramid090);
	vector<Mat> cspyr135 = buildCenterSurroundPyramid(pyramid135);

	//�e�����}�b�v�ɂ��đS�X�P�[�����W��
	Mat_<float> temp(image.size());
	Mat_<float> conspI(image.size(), 0.0f);
	Mat_<float> conspC(image.size(), 0.0f);
	Mat_<float> consp000(image.size(), 0.0f);
	Mat_<float> consp045(image.size(), 0.0f);
	Mat_<float> consp090(image.size(), 0.0f);
	Mat_<float> consp135(image.size(), 0.0f);
	for (int t = 0; t<int(cspyrI.size()); ++t) //CS�s���~�b�h�̊e�w�ɂ���
	{
		//�P�x�����}�b�v�ւ̉��Z
		trimPeaks(cspyrI[t], STEP); resize(cspyrI[t], temp, image.size()); conspI += temp;
		//�F�������}�b�v�ւ̉��Z
		trimPeaks(cspyrRG[t], STEP); resize(cspyrRG[t], temp, image.size()); conspC += temp;
		trimPeaks(cspyrBY[t], STEP); resize(cspyrBY[t], temp, image.size()); conspC += temp;
		//���������}�b�v�ւ̉��Z
		trimPeaks(cspyr000[t], STEP); resize(cspyr000[t], temp, image.size()); consp000 += temp;
		trimPeaks(cspyr045[t], STEP); resize(cspyr045[t], temp, image.size()); consp045 += temp;
		trimPeaks(cspyr090[t], STEP); resize(cspyr090[t], temp, image.size()); consp090 += temp;
		trimPeaks(cspyr135[t], STEP); resize(cspyr135[t], temp, image.size()); consp135 += temp;
	}
	trimPeaks(consp000, STEP);
	trimPeaks(consp045, STEP);
	trimPeaks(consp090, STEP);
	trimPeaks(consp135, STEP);
	Mat_<float> conspO = consp000 + consp045 + consp090 + consp135;

	//�e�����}�b�v���W�񂵌������}�b�v���擾
	trimPeaks(conspI, STEP);
	trimPeaks(conspC, STEP);
	trimPeaks(conspO, STEP);
	Mat saliency = WEIGHT_I*conspI + WEIGHT_C*conspC + WEIGHT_O*conspO;
	normalizeRange(saliency);
	return saliency;
}

int main()
{
	Mat image0 = imread(path);
	imshow("Image", image0);
	waitKey();

	Mat saliency = calcSaliencyMap(image0);
	imshow("saliency", saliency);
	waitKey();
	return 0;
}