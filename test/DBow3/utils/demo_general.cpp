/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <fstream>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector=cv::ORB::create();
    /*if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");*/
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        //cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        //cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        //cout<<"done detecting features"<<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features, vector<string> images)
{
    // branching factor and depth levels
    const int k = 10;
    const int L = 6;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
/*
    DBoW3::Vocabulary voc(k, L, weight, score);

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
    */
    cout << endl << "Loading vocabulary..." << endl;
    DBoW3::Vocabulary voc("small_voc.yml.gz");
    cout << "Done" << endl;
    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    

    BowVector v1, v2;
    ofstream ofs("output.txt");
    for(size_t i = 77607+500*28+1; i <features.size(); i++)
    {
        voc.transform(features[i], v1);
        vector<double> maxs(5,0.0);
        vector<string> name(5);
        cout << "Test number: " << i-77607 << endl;
        for(size_t j =0; j <77608; j++)
        {
            voc.transform(features[j], v2);
            double score = voc.score(v1, v2);
            if (score>maxs[0]) {
                maxs[0]=score;
                name[0]=images[j];
            }
            else if (score>maxs[1]) {
                maxs[1]=score;
                name[1]=images[j];
            }
            else if (score>maxs[2]) {
                maxs[2]=score;
                name[2]=images[j];
            }
            else if (score>maxs[3]) {
                maxs[3]=score;
                name[3]=images[j];
            }
            else if (score>maxs[4]) {
                maxs[4]=score;
                name[4]=images[j];
            }
        }
	ofs << images[i];
        for (int k=0;k<5;k++)
            ofs << ' ' << name[k];
        ofs << endl;
        cout << "Test end" << endl;
    }
    ofs.close();
  
    // save the vocabulary to disk
    
   
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("small_voc.yml.gz");

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 3);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;
    /*
    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
    */
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }
        string line;
        vector<string> images;

        ifstream infile("total_images_paths.txt");
        while (getline(infile, line)) {
            images.push_back(line);
        }
        string descriptor=argv[1];

        //auto images=readImagePaths(argc,argv,2);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);
        testVocCreation(features, images);


        //testDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
