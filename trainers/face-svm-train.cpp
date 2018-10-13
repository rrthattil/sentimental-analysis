//Facial SVM Trainer
#include<iostream>
#include<string>
#include<sstream>
//OpenCV Libraries
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
int main()
{
   string landmark_path="../data/classifiers/landmarks.dat",faces_path="../data/trainer/ckfaces/",emotion_labels[6]={"anger","disgust","happy","nuetral","sadness","surprise"},ext=".png",face_path;
   int face_no[6]={86,87,114,80,64,88},index=0;
   float feature_matrix[519][68*2],label_matrix[519];
   float xlist[68],ylist[68],x[68],y[68],xnormal[68],ynormal[68],xmean,ymean,xsmall,ysmall,xbig,ybig;
   frontal_face_detector detector=get_frontal_face_detector();
   shape_predictor sp;
   deserialize(landmark_path)>>sp;
   array2d<rgb_pixel> img;
   std::vector<full_object_detection> shapes;
   full_object_detection shape;
   for(int i=0;i<6;i++)
   {
       for(int j=0;j<face_no[i];j++)
       {
	   xmean=ymean=0;
	   face_path=faces_path+emotion_labels[i]+"/"+to_string(j+1)+ext;
	   load_image(img,face_path);
	   pyramid_up(img);
	   std::vector<rectangle> dets=detector(img);
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
	   cout<<"The data set for image ("<<emotion_labels[i]<<","<<j+1<<".jpg) is\n";
	   for(int m=0;m<shape.num_parts()*2;m=m+2)
	   {
	       cout<<xnormal[m/2]<<","<<ynormal[m/2]<<"\n";
	       feature_matrix[j+index][m]=xnormal[m/2];
	       feature_matrix[j+index][m+1]=ynormal[m/2];
	   }
       }
	   index=index+face_no[i];
   }
   index=0;
   for(int i=0;i<6;i++)
   {
       for(int j=0;j<519;j++)
       {
	   if(j==index)
	       while(j<index+face_no[i])
	       {
		   label_matrix[j]=1;
		   j++;
	       }
	   else
	       label_matrix[j]=-1;
       }
       index=index+face_no[i];
       Mat labels(519, 1, CV_32SC1, label_matrix);	
       Mat data(519,68*2,CV_32FC1,feature_matrix);
       Ptr<SVM> mymodel=SVM::create();
       mymodel->setType(SVM::C_SVC);
       mymodel->setKernel(SVM::LINEAR);
       mymodel->setC(1);
       mymodel->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-8));
       Ptr<TrainData> td = TrainData::create(data, ROW_SAMPLE, labels);
       for(int i=0;i<10;i++)
	   mymodel->train(td);
       string modelname="../data/classifiers/"+emotion_labels[i]+".xml";
       mymodel->save(modelname);
   }
}
