#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>

using namespace std;
using namespace cv;

const string path = "C:\\bin_OpenCV\\bin\\workspace\\opencv_test\\test_proj\\Debug\\lena.jpg"; //入力画像のパス

const int STEP = 8;
const int GABOR_R = 8; //ガボールカーネルのサイズ（半径）
const float WEIGHT_I = 0.333f; //輝度マップの重み係数
const float WEIGHT_C = 0.333f; //色相マップの重み係数
const float WEIGHT_O = 0.333f; //方向マップの重み係数

//スケールの異なる２つの画像について "center-surround" 演算
Mat operateCenterSurround(const Mat& center, const Mat& surround)
{
	Mat csmap(center.size(), center.type());
	resize(surround, csmap, csmap.size()); //surround画像をcenter画像と同サイズに拡大
	csmap = abs(csmap - center);
	return csmap;
}

//各種ピラミッドから "center-surround" ピラミッドを構築
vector<Mat> buildCenterSurroundPyramid(const vector<Mat>& pyramid)
{
	//surround=center+delta, center={2,3,4}, delta={3,4} 計6枚
	vector<Mat> cspyr(6);
	cspyr[0] = operateCenterSurround(pyramid[2], pyramid[5]);
	cspyr[1] = operateCenterSurround(pyramid[2], pyramid[6]);
	cspyr[2] = operateCenterSurround(pyramid[3], pyramid[6]);
	cspyr[3] = operateCenterSurround(pyramid[3], pyramid[7]);
	cspyr[4] = operateCenterSurround(pyramid[4], pyramid[7]);
	cspyr[5] = operateCenterSurround(pyramid[4], pyramid[8]);
	return cspyr;
}

//画像のダイナミックレンジを[0,1]に正規化
void normalizeRange(Mat& image)
{
	double minval, maxval;
	minMaxLoc(image, &minval, &maxval);

	image -= minval;
	if (minval<maxval)
		image /= maxval - minval;
}

//正規化演算子N(・)：シングルピークの強調とマルチピークの抑制
void trimPeaks(Mat& image, int step)
{
	const int w = image.cols;
	const int h = image.rows;

	const double M = 1.0;
	normalizeRange(image);
	double m = 0.0;
	for (int y = 0; y<h - step; y += step) //端は(h%step)だけ余る
		for (int x = 0; x<w - step; x += step) //端は(w%step)だけ余る
		{
			Mat roi(image, Rect(x, y, step, step));
			double minval = 0.0, maxval = 0.0;
			minMaxLoc(roi, &minval, &maxval);
			m += maxval;
		}
	m /= (w / step - (w%step ? 0 : 1))*(h / step - (h%step ? 0 : 1)); //ブロック数で割って平均を計算
	image *= (M - m)*(M - m);
}

//顕著性マップを計算する
Mat calcSaliencyMap(const Mat& image0)
{
	const Mat_<Vec3f> image = image0 / 255.0f; //ダイナミックレンジの正規化

	//ガボールカーネルの事前生成
	const Size ksize = Size(GABOR_R + 1 + GABOR_R, GABOR_R + 1 + GABOR_R);
	const double sigma = GABOR_R / CV_PI; //±πσまでサポートするように調整
	const double lambda = GABOR_R + 1; //片側を1周期に調整
	const double deg45 = CV_PI / 4.0;
	Mat gabor000 = getGaborKernel(ksize, sigma, deg45 * 0, lambda, 1.0, 0.0, CV_32F);
	Mat gabor045 = getGaborKernel(ksize, sigma, deg45 * 1, lambda, 1.0, 0.0, CV_32F);
	Mat gabor090 = getGaborKernel(ksize, sigma, deg45 * 2, lambda, 1.0, 0.0, CV_32F);
	Mat gabor135 = getGaborKernel(ksize, sigma, deg45 * 3, lambda, 1.0, 0.0, CV_32F);

	const int NUM_SCALES = 9;
	vector<Mat> pyramidI(NUM_SCALES); //輝度ピラミッド
	vector<Mat> pyramidRG(NUM_SCALES); //色相RGピラミッド
	vector<Mat> pyramidBY(NUM_SCALES); //色相BYピラミッド
	vector<Mat> pyramid000(NUM_SCALES); //方向  0°ピラミッド
	vector<Mat> pyramid045(NUM_SCALES); //方向 45°ピラミッド
	vector<Mat> pyramid090(NUM_SCALES); //方向 90°ピラミッド
	vector<Mat> pyramid135(NUM_SCALES); //方向135°ピラミッド

	//特性マップピラミッドの構築
	Mat scaled = image; //最初のスケールは原画像で
	for (int s = 0; s<NUM_SCALES; ++s)
	{
		const int w = scaled.cols;
		const int h = scaled.rows;

		//輝度マップの生成
		vector<Mat_<float> > colors;
		split(scaled, colors);
		Mat_<float> imageI = (colors[0] + colors[1] + colors[2]) / 3.0f;
		pyramidI[s] = imageI;

		//正規化rgb値の計算
		double minval, maxval;
		minMaxLoc(imageI, &minval, &maxval);
		Mat_<float> r(h, w, 0.0f);
		Mat_<float> g(h, w, 0.0f);
		Mat_<float> b(h, w, 0.0f);
		for (int j = 0; j<h; ++j)
			for (int i = 0; i<w; ++i)
			{
				if (imageI(j, i)<0.1f*maxval) //最大ピークの1/10以下の画素は除外
					continue;
				r(j, i) = colors[2](j, i) / imageI(j, i);
				g(j, i) = colors[1](j, i) / imageI(j, i);
				b(j, i) = colors[0](j, i) / imageI(j, i);
			}

		//色相マップの生成（負値は0にクランプ）
		Mat R = max(0.0f, r - (g + b) / 2);
		Mat G = max(0.0f, g - (b + r) / 2);
		Mat B = max(0.0f, b - (r + g) / 2);
		Mat Y = max(0.0f, (r + g) / 2 - abs(r - g) / 2 - b);
		pyramidRG[s] = R - G;
		pyramidBY[s] = B - Y;

		//方向マップの生成
		filter2D(imageI, pyramid000[s], -1, gabor000);
		filter2D(imageI, pyramid045[s], -1, gabor045);
		filter2D(imageI, pyramid090[s], -1, gabor090);
		filter2D(imageI, pyramid135[s], -1, gabor135);

		pyrDown(scaled, scaled); //次のオクターブに向けてスケールダウン
	}

	//center-surround演算
	vector<Mat> cspyrI = buildCenterSurroundPyramid(pyramidI);
	vector<Mat> cspyrRG = buildCenterSurroundPyramid(pyramidRG);
	vector<Mat> cspyrBY = buildCenterSurroundPyramid(pyramidBY);
	vector<Mat> cspyr000 = buildCenterSurroundPyramid(pyramid000);
	vector<Mat> cspyr045 = buildCenterSurroundPyramid(pyramid045);
	vector<Mat> cspyr090 = buildCenterSurroundPyramid(pyramid090);
	vector<Mat> cspyr135 = buildCenterSurroundPyramid(pyramid135);

	//各特性マップについて全スケールを集約
	Mat_<float> temp(image.size());
	Mat_<float> conspI(image.size(), 0.0f);
	Mat_<float> conspC(image.size(), 0.0f);
	Mat_<float> consp000(image.size(), 0.0f);
	Mat_<float> consp045(image.size(), 0.0f);
	Mat_<float> consp090(image.size(), 0.0f);
	Mat_<float> consp135(image.size(), 0.0f);
	for (int t = 0; t<int(cspyrI.size()); ++t) //CSピラミッドの各層について
	{
		//輝度特徴マップへの加算
		trimPeaks(cspyrI[t], STEP); resize(cspyrI[t], temp, image.size()); conspI += temp;
		//色相特徴マップへの加算
		trimPeaks(cspyrRG[t], STEP); resize(cspyrRG[t], temp, image.size()); conspC += temp;
		trimPeaks(cspyrBY[t], STEP); resize(cspyrBY[t], temp, image.size()); conspC += temp;
		//方向特徴マップへの加算
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

	//各特性マップを集約し顕著性マップを取得
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