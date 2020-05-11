#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <algorithm>

#include "dataStructures.h"
#include "structIO.hpp"

using namespace std;

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    /*
    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    double dT = 1 / frameRate;
    TTC = -dT / (1 - meanDistRatio);*/

    // STUDENT TASK (replacement for meanDistRatio)
    double medianDistRatio;
    /*
    //To find median first sort the element
    std::sort(distRatios.begin(), distRatios.end());
    
    unsigned int size = distRatios.size();

    if (size % 2 == 0)
    {
        medianDistRatio =  (distRatios[size / 2 - 1] + distRatios[size / 2]) / 2;
    }
    else
    {
        medianDistRatio = distRatios[size / 2];
    }*/

    size_t n = distRatios.size() / 2;
    std::nth_element(distRatios.begin(), distRatios.begin() + n, distRatios.end());
    medianDistRatio =  distRatios[n];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}

int main()
{
    vector<cv::KeyPoint> kptsSource, kptsRef;
    readKeypoints("../dat/C23A5_KptsSource_AKAZE.dat", kptsSource); // readKeypoints("./dat/C23A5_KptsSource_SHI-BRISK.dat"
    readKeypoints("../dat/C23A5_KptsRef_AKAZE.dat", kptsRef); // readKeypoints("./dat/C23A5_KptsRef_SHI-BRISK.dat"

    vector<cv::DMatch> matches;
    readKptMatches("../dat/C23A5_KptMatches_AKAZE.dat", matches); // readKptMatches("./dat/C23A5_KptMatches_SHI-BRISK.dat", matches);
    double ttc; 
    computeTTCCamera(kptsSource, kptsRef, matches, 10.0, ttc);
    cout << "ttc = " << ttc << "s" << endl;
}