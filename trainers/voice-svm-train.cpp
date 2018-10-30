//Voice SVM Train
#include<iostream>
#include<string>
//For MFCC Extraction
#include<essentia/algorithmfactory.h>
#include<essentia/essentiamath.h>
#include<essentia/pool.h>
#include<math.h>
//For SVM Training
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
//Namespaces
using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace essentia;
using namespace essentia::standard;
int main()
{
    string voices_path="../data/voices/",emotion_labels[7]={"anger","disgust","fear","happy","nuetral","sad","surprise"},ext=".wav",file_path;
    int file_no[7]={62,58,60,60,120,60,60},index=0;
    float feature_matrix[480][2],label_matrix[480];
    for(int i=0;i<7;i++)
    {
       for(int j=0;j<file_no[i];j++)
       {
	   essentia::init();
	   int sampleRate = 44100;
	   int frameSize = 2048;
	   int hopSize = 1024;
	   file_path=voices_path+emotion_labels[i]+"/"+to_string(j+1)+ext;
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
	   feature_matrix[index+j][0]=avg;
	   feature_matrix[index+j][1]=der;
	   cout<<"MFCC Coefficients "<<emotion_labels[i]<<"/"<<j+1<<ext<<"(13): "<<avg<<" & "<<der<<"\n";
	   delete audio;
	   delete fc;
	   delete w;
	   delete spec;
	   delete mfcc;
	   essentia::shutdown(); 
       }
       index+=file_no[i];
   }
   index=0;
   for(int i=0;i<7;i++)
   {
       for(int j=0;j<480;j++)
       {
	   if(j==index)
	       while(j<index+file_no[i])
	       {
		   label_matrix[j]=1;
		   j++;
	       }
	   else
	       label_matrix[j]=-1;
       }
       index=index+file_no[i];
       Mat labels(480, 1, CV_32SC1, label_matrix);	
       Mat data(480,2,CV_32FC1,feature_matrix);
       Ptr<SVM> mymodel=SVM::create();
       mymodel->setType(SVM::C_SVC);
       mymodel->setKernel(SVM::LINEAR);
       mymodel->setC(1);
       mymodel->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-8));
       Ptr<TrainData> td = TrainData::create(data, ROW_SAMPLE, labels);
       for(int i=0;i<10;i++)
	   mymodel->train(td);
       string modelname="../models/voices/"+emotion_labels[i]+".xml";
       mymodel->save(modelname);
   }
} 
