// Dorian SALMI - Mehdi GUITTARD
// Analyse d'image - TP2 Détection de contours
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdlib.h>
#include <time.h>
#include <condition_variable>
#include <stdarg.h>
#include <list>
#include <opencv2/highgui/highgui_c.h>

using namespace std;
using namespace cv;


//------------------------------------------Constants-------------------------------------------------------------------
// Filters
const int PREWIT_HORIZONTAL[3][3] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
};

const int PREWIT_VERTICAL[3][3] = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
};
const int PREWIT_DIAGONAL_1[3][3] = {
        {1, 1, 0},
        {1, 0, -1},
        {0, -1, -1}
};
const int PREWIT_DIAGONAL_2[3][3] = {
        {0, 1, 1},
        {-1, 0, 1},
        {-1, -1, 0}
};

const int SOBEL_HORIZONTAL[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
};
const int SOBEL_VERTICAL[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
};
const int SOBEL_DIAGONAL_1[3][3] = {
        {2, 1, 0},
        {1, 0, -1},
        {0, -1, -2}
};
const int SOBEL_DIAGONAL_2[3][3] = {
        {0, 1, 2},
        {-1, 0, 1},
        {-2, -1, 0}
};

const int KIRSCH_HORIZONTAL[3][3] = {
        {-3, -3, 5},
        {-3, 0, 5},
        {-3, -3, 5}
};
const int KIRSCH_VERTICAL[3][3] = {
        {5, 5, 5},
        {-3, 0, -3},
        {-3, -3, -3}
};
const int KIRSCH_DIAGONAL_1[3][3] = {
        {5, 5, -3},
        {5, 0, -3},
        {-3, -3, -3}
};
const int KIRSCH_DIAGONAL_2[3][3] = {
        {-3, 5, 5},
        {-3, 0, 5},
        {-3, -3, -3}
};

// Colors to mark direction of the gradient
Vec3b color_1 = Vec3b(255,0,0);
Vec3b color_2 = Vec3b(0,255,0);
Vec3b color_3 = Vec3b(0,0,255);
Vec3b color_4 = Vec3b(255,255,255);


//----------------------------------Structure containing convolution results--------------------------------------------
struct Gradient4D{
    float direction_1;
    float direction_2;
    float direction_3;
    float direction_4;

    Gradient4D() {
        this->direction_1 = 0;
        this->direction_2 = 0;
        this->direction_3 = 0;
        this->direction_4 = 0;
    }
};


//---------------------------------Auto threshold-----------------------------------------------------------------------
int autoThreshold(const Mat& img_source) {
    int average_value = 0;
    int nb_pixels = 0;
    for (int i = 0; i <= img_source.rows; i++) {
        for (int j = 0; j <= img_source.cols; j++) {
            Point current = Point (j,i);

            average_value += img_source.at<uchar>(current) ;
            nb_pixels++;
        }
    }
    return average_value/(3*nb_pixels);
}



//----------------------------------Convolution calculation--------------------------------------------------------------
void convolution(const Mat& img_source, const int filter_1[3][3], const int filter_2[3][3], const int filter_3[3][3], const int filter_4[3][3], vector<vector<Gradient4D>>& gradient_table) {
    // Calculate max value of the filter for normalisation
    int max_filter = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (filter_1[i][j] > max_filter) {
                max_filter = filter_1[i][j];
            }
        }
    }

    for (int i = 1; i < img_source.cols-1; i++) {
        for (int j = 1; j < img_source.rows-1; j++) {
            Gradient4D tmp;
            float result_1 = 0;
            float result_2 = 0;
            float result_3 = 0;
            float result_4 = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    tmp.direction_1 += (img_source.at<uchar>(j+k, i+l)*filter_1[k+1][l+1])/max_filter;
                    tmp.direction_2 += (img_source.at<uchar>(j+k, i+l)*filter_2[k+1][l+1])/max_filter;
                    tmp.direction_3 += (img_source.at<uchar>(j+k, i+l)*filter_3[k+1][l+1])/max_filter;
                    tmp.direction_4 += (img_source.at<uchar>(j+k, i+l)*filter_4[k+1][l+1])/max_filter;
                }
            }
            gradient_table[i][j] = tmp;
        }
    }
}


//-------------------------------------Simple threshold-----------------------------------------------------------------
void simpleThreshold2D(Mat& img_result, vector<vector<Gradient4D>>& gradient_table, const int& threshold) {
    int amplitude;
    float slope;
    for (int i = 1; i < img_result.cols-1; i++) {
        for (int j = 1; j < img_result.rows-1; j++) {
            amplitude = sqrt(pow(gradient_table[i][j].direction_1,2) + pow(gradient_table[i][j].direction_1, 2));
            slope = atan(gradient_table[i][j].direction_1/gradient_table[i][j].direction_2);

            if (amplitude > threshold) {
                float  pi = 3.14;
                if (slope > 0 and slope < pi / 5 ) {
                    img_result.at<Vec3b>(Point(i,j)) = color_4;
                } else if (slope < 0  and slope > -pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_2;
                } else if (slope > pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_1;
                } else if (slope < -pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_3;
                }
            }
        }
    }
}

void simpleThreshold4D(Mat& img_result, vector<vector<Gradient4D>>& gradient_table, const int& threshold) {
    float amplitude;
    for (int i = 1; i < img_result.cols-1; i++) {
        for (int j = 1; j < img_result.rows-1; j++) {
            amplitude = max(
                    max(gradient_table[i][j].direction_1, gradient_table[i][j].direction_2),
                    max(gradient_table[i][j].direction_3, gradient_table[i][j].direction_4));
            if (amplitude > threshold) {
                if (amplitude == gradient_table[i][j].direction_1) {
                    img_result.at<Vec3b>(Point(i,j)) = color_1;
                } else if (amplitude == gradient_table[i][j].direction_2) {
                    img_result.at<Vec3b>(Point(i,j)) = color_2;
                } else if (amplitude == gradient_table[i][j].direction_3) {
                    img_result.at<Vec3b>(Point(i,j)) = color_3;
                } else if (amplitude == gradient_table[i][j].direction_4) {
                    img_result.at<Vec3b>(Point(i,j)) = color_4;
                }
            }
        }
    }
}

//-----------------------------------------Hysteresis threshold---------------------------------------------------------
void hysteresisThreshold2D(Mat& img_result, vector<vector<Gradient4D>>& gradient_table) {
    float amplitude;
    float slope;
    float threshold_high = 80;
    float threshold_low = 40;
    list<Point> edge_first_threshold;
    vector<vector<bool>> already_done (img_result.cols, vector<bool>(img_result.rows));
    for (int i = 0; i < already_done.size(); ++i) {
        for (int j = 0; j < already_done[i].size(); ++j) {
            already_done[i][j] = false;
        }
    }

    for (int i = 1; i < img_result.cols-1; i++) {
        for (int j = 1; j < img_result.rows-1; j++) {
            amplitude = sqrt(pow(gradient_table[i][j].direction_1,2) + pow(gradient_table[i][j].direction_1, 2));
            slope = atan(gradient_table[i][j].direction_1/gradient_table[i][j].direction_2);
            if (amplitude > threshold_high) {
                float  pi = 3.14;
                if (slope > 0 and slope < pi / 5 ) {
                    img_result.at<Vec3b>(Point(i,j)) = color_4;
                } else if (slope < 0  and slope > -pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_2;
                } else if (slope > pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_1;
                } else if (slope < -pi/5) {
                    img_result.at<Vec3b>(Point(i,j)) = color_3;
                }
                //img_result.at<Vec3b>(Point(i, j)) = Vec3b(255,255,255);
                edge_first_threshold.push_back(Point(i,j));
                already_done[i][j] = true;
            }
        }
    }

    while(! edge_first_threshold.empty()){
        Point to_check = edge_first_threshold.front();
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (!already_done[to_check.x+i][to_check.y+j]) {
                    amplitude = sqrt(pow(gradient_table[to_check.x + i][to_check.y + j].direction_1, 2) +
                                     pow(gradient_table[to_check.x + i][to_check.y + j].direction_1, 2));
                    slope = atan(gradient_table[to_check.x + i][to_check.y + j].direction_1/gradient_table[to_check.x + i][to_check.y + j].direction_2);

                    if (amplitude > threshold_low) {
                        float  pi = 3.14;
                        if (slope > 0 and slope < pi / 5 ) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_4;
                        } else if (slope < 0  and slope > -pi/5) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_2;
                        } else if (slope > pi/5) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_1;
                        } else if (slope < -pi/5) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_3;
                        }
                        //img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = Vec3b(255, 255, 255);
                        edge_first_threshold.push_back(Point(to_check.x + i, to_check.y + j));
                    }
                    already_done[to_check.x + i][to_check.y + j] = true;
                }
            }
        }
        edge_first_threshold.pop_front();
    }
}

void hysteresisThreshold4D(Mat& img_result, vector<vector<Gradient4D>>& gradient_table) {
    float amplitude;
    float threshold_high = 80;
    float threshold_low = 40;
    list<Point> edge_first_threshold;
    vector<vector<bool>> already_done (img_result.cols, vector<bool>(img_result.rows));
    for (int i = 0; i < already_done.size(); ++i) {
        for (int j = 0; j < already_done[i].size(); ++j) {
            already_done[i][j] = false;
        }
    }

    for (int i = 1; i < img_result.cols-1; i++) {
        for (int j = 1; j < img_result.rows-1; j++) {
            amplitude = max(
                    max(gradient_table[i][j].direction_1, gradient_table[i][j].direction_2),
                    max(gradient_table[i][j].direction_3, gradient_table[i][j].direction_4));
            if (amplitude > threshold_high) {
                if (amplitude == gradient_table[i][j].direction_1) {
                    img_result.at<Vec3b>(Point(i,j)) = color_1;
                } else if (amplitude == gradient_table[i][j].direction_2) {
                    img_result.at<Vec3b>(Point(i,j)) = color_2;
                } else if (amplitude == gradient_table[i][j].direction_3) {
                    img_result.at<Vec3b>(Point(i,j)) = color_3;
                } else if (amplitude == gradient_table[i][j].direction_4) {
                    img_result.at<Vec3b>(Point(i,j)) = color_4;
                }
                //img_result.at<Vec3b>(Point(i,j)) = Vec3b(255,255,255);
                edge_first_threshold.push_back(Point(i,j));
                already_done[i][j] = true;
            }
        }
    }

    while(! edge_first_threshold.empty()){
        Point to_check = edge_first_threshold.front();
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (!already_done[to_check.x+i][to_check.y+j]) {

                    amplitude = max(
                            max(gradient_table[to_check.x + i][to_check.y + j].direction_1,
                                gradient_table[to_check.x + i][to_check.y + j].direction_2),
                            max(gradient_table[to_check.x + i][to_check.y + j].direction_3,
                                gradient_table[to_check.x + i][to_check.y + j].direction_4));
                    if (amplitude > threshold_low) {

                        if (amplitude == gradient_table[to_check.x + i][to_check.y + j].direction_1) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_1;
                        } else if (amplitude == gradient_table[to_check.x + i][to_check.y + j].direction_2) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_2;
                        } else if (amplitude == gradient_table[to_check.x + i][to_check.y +j].direction_3) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_3;
                        } else if (amplitude == gradient_table[to_check.x + i][to_check.y + j].direction_4) {
                            img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = color_4;
                        }
                        //img_result.at<Vec3b>(Point(to_check.x + i, to_check.y + j)) = Vec3b(255, 255, 255);
                        edge_first_threshold.push_back(Point(to_check.x + i, to_check.y + j));
                    }
                    already_done[to_check.x + i][to_check.y + j] = true;
                }
            }
        }
        edge_first_threshold.pop_front();
    }
}



//-------------------------------Display Original and Result image -----------------------------------------------------
void printResults(const Mat& img_source, vector<vector<Mat>> img_results, const int& auto_threshold) {
    const char window_source[] = "Source Image";
    string s = "Auto threshold = " + to_string(auto_threshold) +
            "    -    Left: Kirsch, Center: Sobel, Right: Prewit    -    Top: 2D, Bottom: 4D";
    auto window_result_simple = s.c_str();
    auto window_result_hysteresis = "Hysteresis threshold    -    Left: Kirsch, Center: Sobel, Right: Prewit    -    Top: 2D, Bottom: 4D";

    vector<Mat> tmp_simple (img_results.size());
    vector<Mat> tmp_hysteresis (img_results.size());

    for (int i = 0; i < img_results.size(); i++) {
        vconcat(img_results[i][0], img_results[i][1], tmp_simple[i]);
        vconcat(img_results[i][2], img_results[i][3], tmp_hysteresis[i]);
    }

    Mat result_simple = tmp_simple[0].clone();
    Mat result_hysteresis = tmp_hysteresis[0].clone();

    for (int i = 1; i < img_results.size(); i++) {
        hconcat(result_simple, tmp_simple[i], result_simple);
        hconcat(result_hysteresis, tmp_hysteresis[i], result_hysteresis);
    }


    namedWindow(window_result_simple, WINDOW_AUTOSIZE);
    imshow(window_result_simple, result_simple);

    namedWindow(window_result_hysteresis, WINDOW_AUTOSIZE);
    imshow(window_result_hysteresis, result_hysteresis);

    namedWindow(window_source, WINDOW_AUTOSIZE);
    imshow(window_source, img_source);



    waitKey(0);

    destroyAllWindows();
}

void button_Call_Back () {
    cout << "yay";
}

//---------------------------------------Main------------------------------------------
int main(int argc, char** argv) {
    //-----------------------------------Init------------------------------------------
    srand (time(NULL));

    // Get the path to the image
    string img_path = argv[1]; // Path to our image

    // Object containing the original image
    Mat img_origin;

    // Read the image
    Mat img_source = imread(img_path, IMREAD_COLOR);
    if (img_source.empty()) {
        cout << "Impossible to read the image, check the path" << endl;
        return -1;
    }

    // Copy the image for staging results
    Mat img_result_base = img_source.clone();
    for (int i = 0; i < img_result_base.cols; ++i) {
        for (int j = 0; j < img_result_base.rows; ++j) {
            img_result_base.at<Vec3b>(Point(i, j)) = Vec3b(0, 0, 0);
        }
    } // Fill it with black

    vector<vector<Mat>> img_results(3, vector<Mat>(4));
    for (int i = 0; i < img_results.size(); ++i) {
        for (int j = 0; j < img_results[i].size(); ++j) {
            img_results[i][j] = img_result_base.clone();
        }
    }

    // Make the level of gray image for convolution
    Mat img_gray;
    cvtColor( img_source, img_gray, COLOR_BGR2GRAY );

    // Tables contain
    vector<vector<Gradient4D>> gradient4D_kirsch (img_source.cols, vector<Gradient4D>(img_source.rows));
    vector<vector<Gradient4D>> gradient4D_prewit (img_source.cols, vector<Gradient4D>(img_source.rows));
    vector<vector<Gradient4D>> gradient4D_sobel (img_source.cols, vector<Gradient4D>(img_source.rows));

    // Threshold values for hysteresis threshold
    int threshold_low = -1;
    int threshold_high = -1;



    //--------------------------------Ask for parameters-----------------------------
    cout << "------------------------------Hysteresis threshold-------------------------------" << endl;
    cout << "Processing : " << img_path << endl << endl;

    while (threshold_low < 0) {
        cout << "Low threshold:" << endl;
        cin >> threshold_low;

        if (threshold_low < 0) {
            cout << " !! The low threshold must be above 0." << endl;
        }
    }

    cout << endl;

    while (threshold_high < 0 or threshold_high < threshold_low) {
        cout << "High threshold:" << endl;
        cin >> threshold_high;

        if (threshold_high < 0 or threshold_high < threshold_low) {
            cout << " !! The high threshold must be above 0 and above the low threshold." << endl;
        }
    }



    //--------------------------------Process----------------------------------------
    int auto_threshold = autoThreshold(img_gray);
    cout << "Auto threshold = " << threshold << endl;


    convolution(img_gray, KIRSCH_HORIZONTAL, KIRSCH_VERTICAL,KIRSCH_DIAGONAL_1, KIRSCH_DIAGONAL_2, gradient4D_kirsch);
    simpleThreshold2D(img_results[0][0], gradient4D_kirsch, auto_threshold);
    simpleThreshold4D(img_results[0][1], gradient4D_kirsch, auto_threshold);
    hysteresisThreshold2D(img_results[0][2], gradient4D_kirsch);
    hysteresisThreshold4D(img_results[0][3], gradient4D_kirsch);


    convolution(img_gray, SOBEL_HORIZONTAL, SOBEL_VERTICAL,SOBEL_DIAGONAL_1, SOBEL_DIAGONAL_2, gradient4D_sobel);
    simpleThreshold2D(img_results[1][0], gradient4D_sobel, auto_threshold);
    simpleThreshold4D(img_results[1][1], gradient4D_sobel, auto_threshold);
    hysteresisThreshold2D(img_results[1][2], gradient4D_sobel);
    hysteresisThreshold4D(img_results[1][3], gradient4D_sobel);

    convolution(img_gray, PREWIT_HORIZONTAL, PREWIT_VERTICAL,PREWIT_DIAGONAL_1, PREWIT_DIAGONAL_2, gradient4D_prewit);
    simpleThreshold2D(img_results[2][0], gradient4D_prewit, auto_threshold);
    simpleThreshold4D(img_results[2][1], gradient4D_prewit, auto_threshold);
    hysteresisThreshold2D(img_results[2][2], gradient4D_prewit);
    hysteresisThreshold4D(img_results[2][3], gradient4D_prewit);


    printResults(img_source, img_results, auto_threshold);

    //----------------------------------End---------------------------------------------
    return 0;

}
