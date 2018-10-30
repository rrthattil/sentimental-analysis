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
//Essentia Libraries
#include<essentia/algorithmfactory.h>
#include<essentia/essentiamath.h>
#include<essentia/pool.h>
#include<math.h>
using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace essentia;
using namespace essentia::standard;
float feature_matrix[2];
int main()
{
   string classifier_path="../models/voices/",emotion_labels[7]={"anger","disgust","happy","nuetral","sadness","surprise","fear"},ext=".xml",file_path="../data/temp/voice.wav";
   essentia::init();
   int sampleRate = 44100;
   int frameSize = 2048;
   int hopSize = 1024;
   essentia::standard::AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
   essentia::standard::Algorithm* audio = factory.create("MonoLoader","filename",file_path,"sampleRate",sampleRate);
   essentia::standard::Algorithm* fc    = factory.create("FrameCutter","frameSize",frameSize,"hopSize",hopSize);
   essentia::standard::Algorithm* w     = factory.create("Windowing","type","blackmanharris62");
   essentia::standard::Algorithm* spec  = factory.create("Spectrum");
   essentia::standard::Algorithm* mfcc  = factory.create("MFCC");
   vector<Real> audioBuffer;
   vector<Real> frame, windowedFrame;
   vector<Real> spectrum, mfccCoeffs, mfccBands;
   audio->output("audio").set(audioBuffer);
   fc->input("signal").set(audioBuffer);
   fc->output("frame").set(frame);
   w->input("frame").set(frame);
   w->output("frame").set(windowedFrame);
   spec->input("frame").set(windowedFrame);
   spec->output("spectrum").set(spectrum);
   mfcc->input("spectrum").set(spectrum);
   mfcc->output("bands").set(mfccBands);
   mfcc->output("mfcc").set(mfccCoeffs);   
   audio->compute();
   while(true)
   {
       fc->compute();
       if (!frame.size())
	   break;
       if (isSilent(frame)) 
	   continue;
       w->compute();
       spec->compute();
       mfcc->compute();
   }
   float avg=0.0,der=0.0;
   for(int k=0;k<mfccCoeffs.size();k++)
       avg+=mfccCoeffs.at(k);
   avg/=mfccCoeffs.size();
   for(int k=0;k<mfccCoeffs.size();k++)
       der+=pow(mfccCoeffs.at(k)-avg,2);
   der/=(mfccCoeffs.size()-1);
   der=sqrt(der);
   feature_matrix[0]=avg;
   feature_matrix[1]=der;
   cout<<"MFCC Coefficients: "<<avg<<" & "<<der<<"\n";
   delete audio;
   delete fc;
   delete w;
   delete spec;
   delete mfcc;
   essentia::shutdown(); 
   Mat data(1,2,CV_32F,feature_matrix);
   for(int i=0;i<7;i++)
   {
	string classifier=classifier_path+emotion_labels[i]+ext;
	Ptr<SVM> mymodel=SVM::create();
	mymodel->load(classifier);
	mymodel = StatModel::load<SVM>(classifier);
	float val=mymodel->predict(data);
	if(val>0)
	    return i;
   }
    return 8;
}
