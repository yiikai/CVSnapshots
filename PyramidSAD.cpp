#include <QCoreApplication>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <chrono>


using namespace std;
using namespace cv;
using namespace chrono;

bool matchPointIsOk (int i) { return (i == 1); }
vector<Point2f> srcSelectPoints;
//#define NO_PRAYMAID
void onMouse(int event, int x, int y, int flags, void* param)
{
    Mat *im = reinterpret_cast<Mat*>(param);
    switch(event)
    {
        case EVENT_LBUTTONDOWN:
        {
            cout<<"push"<<endl;
            srcSelectPoints.push_back(Point(x,y));
        }break;
        default:break;
    }
}

Point2f computerFullSAD(Mat& tmp, Mat& dst)
{
    int width = dst.cols;
    int height = dst.rows;
    int x=0,y=0,minval = INT_MAX;
    for(int i = 0; i < width - tmp.cols; i++)
    {
        for(int j = 0; j < height - tmp.rows; j++)
        {
            Mat diff;
            absdiff(tmp, dst(Rect(i,j,tmp.cols,tmp.rows)),diff);
            Scalar Add = sum(diff);
            float add = Add[0];
            if(add < minval)
            {
                x = i;
                y = j;
                minval = add;
            }
        }
    }
    return Point2f(x,y);
}

Point2f computeSAD(Mat& tmp, Mat& dst, Point2f lastLevelPoints, int expend)
{
    // last level points * 2 to meet next level compute SAD
    int x=0,y=0,minval = INT_MAX;
    Point2f points = lastLevelPoints * 2;
    int mincols = points.x - expend;
    int maxcols = points.x + expend;
    int minrows = points.y - expend;
    int maxrows = points.y + expend;
    for(int i = mincols; i < maxcols; i++)
    {
        for(int j = minrows; j < maxrows; j++)
        {
            Mat selectdst = dst(Rect2f(i,j,tmp.cols,tmp.rows));
            Mat diff;
            absdiff(tmp,selectdst,diff);
            Scalar Add = sum(diff);
            float add = Add[0];
            if(add < minval)
            {
                x = i;
                y = j;
                minval = add;
            }
        }
    }
    return Point2f(x,y);

}

Point2f computePyramidsMatch(vector<Mat>& tmpPyram, vector<Mat>& dstPyram, int levels)
{
    Point2f selectPts;
    for(int i = 0; i < levels; i++)
    {
        if(i == 0)
        {
            selectPts = computerFullSAD(tmpPyram[levels - i - 1], dstPyram[levels - i -1]);
        }
        else
        {
            selectPts = computeSAD(tmpPyram[levels - i - 1], dstPyram[levels - i -1], selectPts, 7);
        }
    }
    return selectPts;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    Mat src = imread("/Users/yiikai/Desktop/Cplus_plus.jpeg",COLOR_BGR2GRAY);
    Mat dst = imread("/Users/yiikai/Desktop/Cplus_plus.jpeg",COLOR_BGR2GRAY);
    namedWindow("src");
    setMouseCallback("src",onMouse,reinterpret_cast<void*>(&src));
    imshow("src",src);
    waitKey();
#if defined NO_PRAYMAID
    cout<<"select points num: "<<srcSelectPoints.size()<<endl;
    int width = srcSelectPoints[1].x - srcSelectPoints[0].x;
    int height = srcSelectPoints[2].y - srcSelectPoints[0].y;
    Rect2f srcwindow(srcSelectPoints[0].x,srcSelectPoints[0].y,width,height);
    Mat Kernel_L = src(srcwindow);
    int minx = 0,miny = 0,minval = INT_MAX;
    auto start = system_clock::now();
    for(int i = 0; i < dst.cols - width; i++)
    {
        for(int j = 0; j < dst.rows - height;j++)
        {
            Mat Kernel_R = dst(Rect2f(i,j,width,height));
            Mat diff;
            absdiff(Kernel_L,Kernel_R,diff);
            Scalar Add = sum(diff);
            float val = Add[0];
            if(val < minval)
            {
                minx = i;
                miny = j;
                minval = val;
                cout<<"minval: "<<minval<<endl;
            }
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout<<"use time: "<< duration.count()<<"ms"<<endl;
    Mat finimg = dst(Rect2f(minx,miny,width,height));
    imshow("dst",finimg);
    waitKey(0);
#else
    //do praymid match
    int width = srcSelectPoints[1].x - srcSelectPoints[0].x;
    int height = srcSelectPoints[2].y - srcSelectPoints[0].y;
    Rect2f srcwindow(srcSelectPoints[0].x,srcSelectPoints[0].y,width,height);
    Mat tmpImg = src(srcwindow);
    int pryLevels = 4;
    vector<Mat> tmpPyramids, dstPyramids;
    Mat tmp_0;
    tmpImg.copyTo(tmp_0);
    tmpPyramids.push_back(tmpImg);
    for(int i = 0; i < pryLevels; i++)
    {
        Mat lowerImg;
        pyrDown(tmp_0, lowerImg);
        tmpPyramids.push_back(lowerImg);
        tmp_0 = lowerImg;
    }
    Mat dst_0;
    dst.copyTo(dst_0);
    dstPyramids.push_back(dst);
    for(int i = 0; i < pryLevels; i++)
    {
        Mat lowerImg;
        pyrDown(dst_0,lowerImg);
        dstPyramids.push_back(lowerImg);
        dst_0 = lowerImg;
    }
    auto start = system_clock::now();
    Point2f finalpts = computePyramidsMatch(tmpPyramids, dstPyramids, pryLevels+1);
    auto end = system_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout<<"use time: "<< duration.count()<<"ms"<<endl;
    Mat finalimg = dst(Rect2f(finalpts.x,finalpts.y,width,height));
    imshow("dst",finalimg);
    waitKey();

#endif
    return a.exec();
}
