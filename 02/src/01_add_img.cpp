#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace cv;

int main()
{
    Mat h_img1 = imread("../img/rain.jpg");
    Mat h_img2 = imread("../img/mushroom.jpg");
    cuda::GpuMat d_img1, d_img2, d_img_sum;
    d_img1.upload(h_img1);  // 将图片上传至GPU
    d_img2.upload(h_img2);
    cuda::addWeighted(d_img1, 0.7, d_img2, 0.3, 0, d_img_sum);
    Mat h_img_sum;
    d_img_sum.download(h_img_sum);
    imshow("0", h_img_sum);
    waitKey(0);
    return 0;
}