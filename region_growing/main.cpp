#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <condition_variable>
#include <list>

using namespace std;
using namespace cv;


// ---------------------------------Constants--------------------------------------------
const int NUMBER_REGIONS_X = 10;
const int NUMBER_REGIONS_Y = 10;

// The points to check relative to the point we are currently considering
const Point SURROUNDINGS[8] = {
        Point (-1, 0),
        Point (0, -1),
        Point (1, 0),
        Point (0, 1),
        Point (-1, 1),
        Point (1, -1),
        Point (1, 1),
        Point (-1, -1)
};

mutex m;

float MAX_COLOR = 0;

float MAXCOLOR=0;



//-------------------------------Generate Random Color------------------------------------
Vec3b makeRandomColor() {
    return Vec3b (
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255,
            (double)std::rand() / RAND_MAX * 255
    );
}



//-----------------------------Distance between two points---------------------------
float distanceBetweenTwoPoints(const Mat& img,const Point& origin , const Point& test) {
    Vec3b color_origin = img.at<Vec3b>(origin);
    Vec3b color_test = img.at<Vec3b>(test);
    return sqrt(pow(((float)color_origin[0]-(float)color_test[0]),2.f)+pow(((float)color_origin[1]-(float)color_test[1]),2.f)+pow(((float)color_origin[2]-(float)color_test[2]),2.f));
}


//----------------------------Calculate Average color of a point--------------------------------------------------------
float averageColor(const Mat& img,const Point& origin) {
    Vec3b color_origin = img.at<Vec3b>(origin);
    return ((float)color_origin[0]+(float)color_origin[1]+(float)color_origin[2])/3.;
}


//------------------------------------Struct that modelize a region-----------------------------------------------------
struct Region {
    int id;
    vector<Point> content;
    vector<int> neighbors;
    float average;
    Vec3b region_color;

    Region(const int& id_region) {
        this->id = id_region;
        this->average = 0;
        this->region_color = makeRandomColor();
    }

    void calculateAverage(const Mat& img_source) {
        float tmp = 0;
        for (int j = 0; j < this->content.size(); j++) {
            tmp += averageColor(img_source, this->content[j]);
        }
        this->average = tmp / this->content.size();
    }
};




//------------------------------Region Growing (grows one region)------------------------------------------
void regionGrow(const Mat& img_source, vector<vector<int>>& regions, vector<Region>& list_regions, Point& seed, int& included_pixels, const int& threshold_growing) {
    list<Point> to_visit;
    to_visit.push_back(seed);

    while (!to_visit.empty()) {
        Point current_center = to_visit.front();
        to_visit.pop_front();

        for (int i = 0; i < 8; i++) {
            Point current_check = current_center + SURROUNDINGS[i];
            // Checks if the point is within the image
            if (current_check.x >= 0 && current_check.x < img_source.cols
                && current_check.y >= 0 && current_check.y < img_source.rows) {
                if (regions[current_check.x][current_check.y] == -1) {
                    float distance = distanceBetweenTwoPoints(img_source, current_check, current_center);
                    int region_current_id = regions[current_center.x][current_center.y];
                    if (distance < threshold_growing) {
                        m.lock();
                        regions[current_check.x][current_check.y] = region_current_id;
                        to_visit.push_back(current_check);
                        list_regions[region_current_id].content.push_back(current_check);
                        //list_regions[region_current_id].calculateAverage(img_source);
                        included_pixels++;
                        m.unlock();
                    }
                } else if(regions[current_check.x][current_check.y] != regions[current_center.x][current_center.y]) {
                        m.lock();
                        // Adds the encountered region in the adjacency graph for both points
                        list_regions[regions[current_check.x][current_check.y]].neighbors.push_back(regions[current_center.x][current_center.y]);
                        list_regions[regions[current_center.x][current_center.y]].neighbors.push_back(regions[current_check.x][current_check.y]);
                        m.unlock();
                }
            }
        }
    }
}




//---------------------------------Treatment function------------------------------------------
void regionGrowAll(const Mat& img_source, vector<vector<int>> regions, vector<Region>& list_regions, const int& threshold_growing) {
    // Number of pixels in the image
    int total_pixels = img_source.cols * img_source.rows;

    // Number of pixels inside a region
    int included_pixels = 0;

    // Counter of the number of regions
    int nb_region = 0;

    // Table of node for the adjacency graph
    vector<thread> threads;
    vector<Point> seeds;
    int slice_x = img_source.cols / NUMBER_REGIONS_X;
    int slice_y = img_source.rows / NUMBER_REGIONS_Y;
    for (int i = 0; i < NUMBER_REGIONS_X; i++) {
        for (int j = 0; j < NUMBER_REGIONS_Y; j++) {
            int seed_x = rand()%slice_x + (i * slice_x);
            int seed_y = rand()%slice_y + (j * slice_y);
            Point seed = Point(seed_x, seed_y);
            seeds.push_back(seed);
            list_regions.push_back(Region(nb_region));
            list_regions[nb_region].content.push_back(seed);
            regions[seed_x][seed_y] = nb_region;
            nb_region++;
            included_pixels ++;
        }
    }


    // Adding more seeds where the pixels have undetermined regions
    while (included_pixels < total_pixels){
        if(seeds.empty()){
            for (int i = 0; i < img_source.cols; i++) {
                for (int j = 0; j < img_source.rows; j++) {
                    if (included_pixels < total_pixels && regions[i][j] == -1 && rand()%100 < 5) {
                            Point seed = Point(i, j);
                            seeds.push_back(seed);
                            list_regions.push_back(Region(nb_region));
                            list_regions[nb_region].content.push_back(seed);
                            regions[i][j] = nb_region;
                            nb_region++;
                            included_pixels ++;
                    }
                }
            }
        }
        if (!seeds.empty()) {
            for(int i = 0; i<seeds.size();i++) {
                threads.push_back(thread(regionGrow, ref(img_source), ref(regions), ref(list_regions), ref(seeds[i]), ref(included_pixels), ref(threshold_growing)));
            }
            seeds.clear();
            if(!threads.empty()) {
                for(int i = 0; i<threads.size();i++) {
                    threads[i].join();
                }
                threads.clear();
            }
        }
    }

    // Simplify the adjacency graph (sorting and deleting the  duplicate values)
    for (int i = 0; i < list_regions.size(); i++) {
        sort(list_regions[i].neighbors.begin(), list_regions[i].neighbors.end() );
        list_regions[i].neighbors.erase(unique(list_regions[i].neighbors.begin(), list_regions[i].neighbors.end() ), list_regions[i].neighbors.end() );
    }
}



//--------------------------------------Fusion of the regions------------------------------------------------------------
void regionFusion(const Mat& img_source, vector<Region>& list_regions, const int& threshold_fusion) {
    for (int i = 0; i < list_regions.size(); i++) {
        list_regions[i].calculateAverage(img_source);
    }

    for (int m = 0; m < 2; m++) {
        for (int i = 0; i < list_regions.size(); i++) {
            while (!list_regions[i].neighbors.empty()) {
                int id_checking = list_regions[i].neighbors[0];
                if (i != id_checking && abs(list_regions[i].average - list_regions[id_checking].average) < threshold_fusion) {
                    for (int j = 0; j < list_regions[id_checking].neighbors.size(); j++) {
                        if (list_regions[id_checking].neighbors[j] != i) {
                            list_regions[i].neighbors.insert(list_regions[i].neighbors.end(),
                                                             list_regions[id_checking].neighbors[j]);
                        }
                    }
                    list_regions[i].content.insert(list_regions[i].content.end(),
                                                   list_regions[list_regions[i].neighbors[0]].content.begin(),
                                                   list_regions[list_regions[i].neighbors[0]].content.end());
                    list_regions[id_checking].content.clear();
                    list_regions[id_checking].neighbors.clear();
                }
                list_regions[i].neighbors.erase(list_regions[i].neighbors.begin());
                list_regions[i].calculateAverage(img_source);
            }
        }
    }
}



//------------------------Colors the result image--------------------------
void colorResult(Mat& img_result, const vector<Region>& list_regions) {
    for (int i = 0; i < list_regions.size(); i++) {
        for (int j = 0; j < list_regions[i].content.size(); j++) {
            img_result.at<Vec3b>(list_regions[i].content[j]) = list_regions[i].region_color;
        }
    }
}


//------------------------Draws borders on the result image--------------------------
void borderResult(Mat& img_result,  vector<vector<int>>& regions, const vector<Region> list_regions) {
    // Update region
    for (int i = 0; i < list_regions.size(); i++) {
        for (int j = 0; j < list_regions[i].content.size(); j++) {
            Point current = list_regions[i].content[j];
            regions[current.x][current.y] = list_regions[i].id;
        }
    }

    // Detection of borders
    vector<vector<int>> front(
            img_result.cols,
            vector<int>(img_result.rows)
    );
    for (int i = 0; i < img_result.cols-1; i++) {
        for (int j = 0; j < img_result.rows-1; j++) {
            if(regions[i][j] != regions[i][j+1]) {
                front[i][j] = 1;
                front[i][j+1] = 1;
            }
            if(regions[i][j] != regions[i+1][j]) {
                front[i][j] = 1;
                front[i+1][j] = 1;
            }
            if(regions[i][j] != regions[i+1][j+1]) {
                front[i][j] = 1;
                front[i+1][j+1] = 1;
            }
        }
    }

    Vec3b cols_front(0,0,0);
    Vec3b background(255,255,255);
    for (int i = 0; i < img_result.cols; ++i) {
        for (int j = 0; j < img_result.rows; ++j) {
            if (front[i][j] !=0) {
                img_result.at<Vec3b>(Point(i, j)) = cols_front;
            }else {
                img_result.at<Vec3b>(Point(i, j)) = background;
            }

        }
    }
}




//-------------------------------Display Original and Result image with both borders and colors------------------------
void printResults(const Mat& img_source, const Mat& img_result_color, const Mat& img_result_border) {
    const char window_source[] = "Source Image";
    const char window_result_color[] = "Result Image with color";
    const char window_result_border[] = "Result Image with borders";

    namedWindow(window_source, WINDOW_AUTOSIZE);
    imshow(window_source, img_source);

    namedWindow(window_result_color, WINDOW_AUTOSIZE);
    imshow(window_result_color, img_result_color);
    moveWindow(window_result_color, 200, 200);

    namedWindow(window_result_border, WINDOW_AUTOSIZE);
    imshow(window_result_border, img_result_border);
    moveWindow(window_result_border, 400, 400);

    waitKey(0);

    destroyAllWindows();
}



//---------------------------------------MAIN-------------------------------------------
int main(int argc, char** argv)
{
    //-----------------------------------Init------------------------------------------
    srand (time(NULL));

    // The threshold values
    float threshold_growing;
    float threshold_fusion;

    // Get the path to the image
    string img_path = argv[1]; // Path to our image

    // Read the image
    Mat img_source = imread(img_path, IMREAD_COLOR);
    if (img_source.empty()) {
        cout << "Impossible to read the image, check the path" << endl;
        return -1;
    }

    // Copy the image for staging results
    Mat img_result_border = img_source.clone();
    Mat img_result_color = img_source.clone();

    // List of regions (see struct above for details)
    vector<Region> list_regions;

    // Table tracking region of each pixels (-1 : no region) useful for the region growing
    vector<vector<int>> regions(
            img_source.cols,
            vector<int>(img_source.rows)
    );
    for (int i = 0; i < img_source.cols; ++i) {
        for (int j = 0; j < img_source.rows; ++j) {
            regions[i][j] = -1;
        }
    }

    //--------------------------------Ask for parameters-----------------------------
    cout << "------------------------------Region Growing-------------------------------" << endl;
    cout << "Processing : " << img_path << endl;

    cout << "Enter the threshold for the region grow :" << endl;
    cin >> threshold_growing;

    cout << "Enter the threshold for the fusion of the regions:" << endl;
    cin >> threshold_fusion;



    //--------------------------------Process----------------------------------------

    // Calculates mean and deviation of the source image
    //Scalar mean, deviation;
    //meanStdDev(img_source, mean, deviation);
    //MAX_COLOR = (deviation[0] + deviation[1] + deviation[2])/3.;



    // Region Grow
    auto start = chrono::high_resolution_clock::now();
    regionGrowAll(img_source, regions, list_regions, threshold_growing);
    auto end = chrono::high_resolution_clock::now();
    auto int_s = chrono::duration_cast<chrono::milliseconds>(end-start);
    cout << "Region growing : " << int_s.count() << " ms" << endl;


    // Region Fusion
    start = chrono::high_resolution_clock::now();
    regionFusion(img_source, list_regions, threshold_fusion);
    end = chrono::high_resolution_clock::now();
    int_s = chrono::duration_cast<chrono::milliseconds>(end-start);
    cout << "Region Fusion : " << int_s.count() << " ms" << endl;


    // Filling result images (borders and colors)
    start = chrono::high_resolution_clock::now();
    colorResult(img_result_color, list_regions);
    borderResult(img_result_border, regions, list_regions);
    int_s = chrono::duration_cast<chrono::milliseconds>(end-start);
    cout << "Filling result images : " << int_s.count() << " ms" << endl;


    printResults(img_source, img_result_color, img_result_border);

//----------------------------------End---------------------------------------------
    return 0;
}