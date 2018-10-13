//Face Detection
#include<iostream>
#include<time.h>
#include<string>
#include<sstream>
//OpenCV Libraries
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
//DLib Libraries
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/image_processing/render_face_detections.h>
#include<dlib/image_processing.h>
#include<dlib/gui_widgets.h>
#include<dlib/image_io.h>
#define DLIB_PNG_SUPPORT
using namespace std;
using namespace dlib;
using namespace cv;
using namespace cv::ml;
int detect(Mat& frame,CascadeClassifier& cascade);
int main()
{
   string landmark_path="../data/classifiers/landmarks.dat",cascade_path="../data/classifiers/face.xml",classifier_path="../data/classifiers/",emotion_labels[6]={"anger","disgust","happy","nuetral","sadness","surprise"},ext=".xml",face_path="../data/temp/face.png";
   float x[68],y[68],xlist[68],ylist[68],xnormal[68],ynormal[68],xmean=0,ymean=0,xbig,ybig,xsmall,ysmall,feature_matrix[68*2];
    int flag=0;
    frontal_face_detector detector=get_frontal_face_detector();
    shape_predictor sp;
    deserialize(landmark_path)>>sp;
    array2d<rgb_pixel> img;
    std::vector<full_object_detection> shapes;
    full_object_detection shape;
    VideoCapture capture;
    Mat frame, image;
    CascadeClassifier face;
    face.load(cascade_path);
    capture.open(0);
    if(capture.isOpened())
    {
	//clock_t start=clock();
	cout<<"Hello there.Wait till I analyse your face"<<endl;
	//while(flag==0||(clock()-start)/CLOCKS_PER_SEC>10)
	//{
	    capture>>frame;
	 //   if(frame.empty())
	//	break;
	//    Mat f=frame.clone();
	//    flag=detect(f,face);
	//}
	//if(flag==0)
	 //   cout<<"No person detected"<<endl;
	//else
	    imwrite(face_path,frame);
    }
    else
	cout<<"No Camera"<<endl;
    load_image(img,face_path);
    pyramid_up(img);
    std::vector<dlib::rectangle> dets=detector(img);
    for(int m=0;m<dets.size();m++)
    {
	shape=sp(img,dets[m]);
	shapes.push_back(shape);
    }
    for(int m=0;m<shape.num_parts();m++)
    {
	xlist[m]=shape.part(m).x();
	xmean+=xlist[m];
	ylist[m]=shape.part(m).y();
	ymean=+ylist[m];
    }
    xmean/=68;
    ymean/=68;
    for(int m=0;m<shape.num_parts();m++)
    {
	x[m]=xlist[m]-xmean;
	y[m]=ylist[m]-ymean;
    }
    xsmall=xbig=x[0];
    ysmall=ybig=y[0];
    for(int m=0;m<shape.num_parts();m++)
    {
	if(x[m]>xbig)
	    xbig=x[m];
	if(x[m]<xsmall)
	    xsmall=x[m];
	if(y[m]>ybig)
	    ybig=y[m];
	if(y[m]>ybig)
	    ybig=y[m];
    }
    for(int m=0;m<shape.num_parts();m++)
    {
	xnormal[m]=(x[m]-xsmall)/(xbig-xsmall);
	ynormal[m]=(y[m]-ysmall)/(ybig-ysmall);
    }
    //cout<<"The Points are:\n";
    for(int m=0;m<shape.num_parts()*2;m=m+2)
    {
	//cout<<xnormal[m/2]<<","<<ynormal[m/2]<<"\n";
	feature_matrix[m]=xnormal[m/2];
	feature_matrix[m+1]=ynormal[m/2];
    }
    Mat data(1,68*2,CV_32F,feature_matrix);
    for(int i=0;i<6;i++)
    {
	string classifier=classifier_path+emotion_labels[i]+ext;
	Ptr<SVM> mymodel=SVM::create();
	mymodel->load(classifier);
	mymodel = StatModel::load<SVM>(classifier);
	float val=mymodel->predict(data);
	if(val>0)
	    return i;
    }
    return 6;
}
int detect(Mat& frame,CascadeClassifier& cascade)
{
    std::vector<Rect> faces;
    Mat g1,g2;
    int f=0;
    char name;
    string path("../data/temp/faces/"),ext(".png"),file;
    cvtColor(frame,g1,COLOR_BGR2GRAY);
    resize(g1,g2,Size(),1,1,INTER_LINEAR);
    equalizeHist(g2,g2);
    cascade.detectMultiScale(g2,faces,1.1,2,0|CASCADE_SCALE_IMAGE, Size(30, 30));
    if(faces.size()>=1) return f;
    else return 0;
}

