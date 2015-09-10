
// OpenCV offline
#include "flycapture/FlyCapture2.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <PupilTracker.h>
#include <cvx.h>
#include <utils.h>
#include <tbb/tbb.h>
#include <random>
#include <boost/foreach.hpp>
#include <boost/circular_buffer.hpp>

#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <cstring>

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <sys/socket.h>
#include <netdb.h>
#include <comedilib.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
#include <ctype.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <cstdlib>
#include <cmath>
#include <netdb.h>


using namespace FlyCapture2;
using namespace cv;
using namespace std;

// Initialize
comedi_t *devx;
comedi_t *devy;
#define SAMPLE_CT 5 // about as short as you can get
#define BUF_LEN 0x8000
#define COMEDI_DEVICE_AO "/dev/comedi0"

// extra feature crap
float centerX = 0;
float centerY = 0;

int center_offset_x = 0;
int center_offset_y = 0;
int max_rngx;
int max_rngy;

Vec<float,2> offset;

// initialize positions
double xpos = 0;
double ypos = 0;

// initialize initial value of input image
float xmax;
float ymax;

double amplitude_x = 4000;
double amplitude_y = 4000;
double fudge_x = 2048;
double fudge_y = 2048;

const char *comdevice = COMEDI_DEVICE_AO;
const char *comdevice2 = COMEDI_DEVICE_AO;

int external_trigger_number = 0;

lsampl_t dataxl[SAMPLE_CT];
lsampl_t datayl[SAMPLE_CT];

struct parsed_options_hold{
  char filename[256];
  double value;
  int subdevice;
  int channel;
  int aref;
  int range;
  int physical;
  int verbose;
  int n_chan;
  int n_scan;
  double freq;
};


char *cmd_src(int src,char *buf)
{
        buf[0]=0;

        if(src&TRIG_NONE)strcat(buf,"none|");
        if(src&TRIG_NOW)strcat(buf,"now|");
        if(src&TRIG_FOLLOW)strcat(buf, "follow|");
        if(src&TRIG_TIME)strcat(buf, "time|");
        if(src&TRIG_TIMER)strcat(buf, "timer|");
        if(src&TRIG_COUNT)strcat(buf, "count|");
        if(src&TRIG_EXT)strcat(buf, "ext|");
        if(src&TRIG_INT)strcat(buf, "int|");
        if(src&TRIG_OTHER)strcat(buf, "other|");

        if(strlen(buf)==0){
                sprintf(buf,"unknown(0x%08x)",src);
        }else{
                buf[strlen(buf)-1]=0;
        }

        return buf;
}


// Comedi script for 2 channel analog output
int comedi_internal_trigger_cust(comedi_t* device, int subdevice, int channelx, int channely, lsampl_t* dataxl, lsampl_t* datayl, int range, int aref)
{

  comedi_insn insn[2];
  comedi_insnlist il;

  il.n_insns=2;
  il.insns=insn;

  memset(&insn[0], 0, sizeof(comedi_insn));
  insn[0].insn = INSN_WRITE; //INSN_INTTRIG
  insn[0].subdev = subdevice;
  insn[0].data = dataxl;
  insn[0].n = SAMPLE_CT;
  insn[0].chanspec = CR_PACK(channelx,range,aref);

  memset(&insn[1], 0, sizeof(comedi_insn));
  insn[1].insn = INSN_WRITE; //INSN_INTTRIG
  insn[1].subdev = subdevice;
  insn[1].data = datayl;
  insn[1].n = SAMPLE_CT;
  insn[1].chanspec = CR_PACK(channely,range,aref);

  return comedi_do_insnlist(device, &il);
}


void dump_cmd(FILE *out,comedi_cmd *cmd)
{
        char buf[10];

        fprintf(out,"subdevice:      %d\n",
                cmd->subdev);

        fprintf(out,"start:      %-8s %d\n",
                cmd_src(cmd->start_src,buf),
                cmd->start_arg);

        fprintf(out,"scan_begin: %-8s %d\n",
                cmd_src(cmd->scan_begin_src,buf),
                cmd->scan_begin_arg);

        fprintf(out,"convert:    %-8s %d\n",
                cmd_src(cmd->convert_src,buf),
                cmd->convert_arg);

        fprintf(out,"scan_end:   %-8s %d\n",
                cmd_src(cmd->scan_end_src,buf),
                cmd->scan_end_arg);

        fprintf(out,"stop:       %-8s %d\n",
                cmd_src(cmd->stop_src,buf),
                cmd->stop_arg);
}



// tcpclient:
// A class that creates a socket to allow communication between machines
// This allows streaming data to another machine
// KEEP THIS AROUND FOR YOUR OWN GOOD!
class tcpclient{
private:
	int status;
	struct addrinfo host_info;
	struct addrinfo *host_info_list;
	int socketfd;
	const char *msg;
	int len;
	ssize_t bytes_sent;
	ssize_t bytes_recieved;
	char incoming_data_buffer[100];


public:
	void initialize(const char* hostname, const char* port){
		// need to block out memory and set to 0s
		memset(&host_info, 0, sizeof host_info);
		std::cout << "Setting up structs..." << std::endl;
		host_info.ai_family = AF_UNSPEC;
		host_info.ai_socktype = SOCK_STREAM;
		status = getaddrinfo(hostname, port, &host_info, &host_info_list);
		if (status != 0) std::cout << "getaddrinfo error" << gai_strerror(status);

		std::cout << "Creating a socket... " << std::endl;
		socketfd = socket(host_info_list->ai_family, host_info_list->ai_socktype, host_info_list->ai_protocol);
		if (socketfd == -1) std::cout << "Socket error";

		std::cout << "Connecting..." << std::endl;
		status = connect(socketfd, host_info_list->ai_addr, host_info_list->ai_addrlen);
		if (status == -1) std::cout << "Connect error";
	}
};

// CStopWatch:
// A simple timer class with Start, Stop, and GetDuration function calls
class CStopWatch{
private:
	clock_t start;
	clock_t finish;

public:
	double GetDuration() {return (double)(finish-start) / CLOCKS_PER_SEC;}
	void Start() {start = clock();}
	void Stop() {finish = clock();}

};


// Initialize global variables: These are necessary for GUI function
int max_solves_slider_max = 100;
int max_solves_slider;
int max_solves = 100;

Vec<int,4> coordinates;
float tracking_params[] = {0, 0, 0, 0};

int min_dist;

int min_radius_slider_max = 199;
int min_radius_slider;
int min_radius;

int max_radius_slider_max = 200;
int max_radius_slider;
int max_radius;

int canny_threshold1_slider_max = 100;
int canny_threshold1_slider;
int canny_threshold1;

int canny_threshold2_slider_max = 100;
int canny_threshold2_slider;
int canny_threshold2;

int canny_blur_slider_max = 10;
int canny_blur_slider;
int canny_blur;

int starburst_pt_slider_max = 100;
int starburst_pt_slider;
int starburst_pt;

int rec_slider_max = 1;
int rec_slider;
int record_video;

int orientation_slider_max = 1;
int orientation_slider;
int orientation;

int centerx_slider = 0;
int centerx = 0;

int centery_slider = 0;
int centery = 0;

int offsetx_slider_max = 20;
int offsetx_slider = cvFloor(offsetx_slider_max*0.5);
int offsetx = 0;

int offsety_slider_max = 20;
int offsety_slider = cvFloor(offsety_slider_max*0.5);
int offsety = 0;

int video_display_slider_max = 1;
int video_display_slider;
int video_display;

int run_program_slider_max = 1;
int run_program_slider = 1;
int run_program = 1;

int save_csv_slider_max = 1;
int save_csv_slider = 0;
int save_csv = 0;

int stream_data_slider_max = 1;
int stream_data_slider = 0;
int stream_data = 0;

int downsample_slider_max = 1000;
int downsample_slider = 1;
int downsample = 1;

// ellipse fitting utils
static std::mt19937 static_gen;
int pupiltracker::random(int min, int max)
{
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(static_gen);
}
int pupiltracker::random(int min, int max, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(gen);
}

// ellipse fitting utils
void pupiltracker::cvx::getROI(const cv::Mat& src, cv::Mat& dst, const cv::Rect& roi, int borderType)
{
    cv::Rect bbSrc = boundingBox(src);
    cv::Rect validROI = roi & bbSrc;
    if (validROI == roi)
    {
        dst = cv::Mat(src, validROI);
    }
    else
    {
        // Figure out how much to add on for top, left, right and bottom
        cv::Point tl = roi.tl() - bbSrc.tl();
        cv::Point br = roi.br() - bbSrc.br();

        int top = std::max(-tl.y, 0);  // Top and left are negated because adding a border
        int left = std::max(-tl.x, 0); // goes "the wrong way"
        int right = std::max(br.x, 0);
        int bottom = std::max(br.y, 0);

        cv::Mat tmp(src, validROI);
        cv::copyMakeBorder(tmp, dst, top, bottom, left, right, borderType);
    }
}


float pupiltracker::cvx::histKmeans(const cv::Mat_<float>& hist, int bin_min, int bin_max, int K, float init_centres[], cv::Mat_<uchar>& labels, cv::TermCriteria termCriteria)
{
    CV_Assert( hist.rows == 1 || hist.cols == 1 && K > 0 );

    labels = cv::Mat_<uchar>::zeros(hist.size());
    int nbins = hist.total();
    float binWidth = (bin_max - bin_min)/nbins;
    float binStart = bin_min + binWidth/2;

    cv::Mat_<float> centres(K, 1, init_centres, 4);

    int iters = 0;
    bool finalRun = false;
    while (true)
    {
        ++iters;
        cv::Mat_<float> old_centres = centres.clone();

        int i_bin;
        cv::Mat_<float>::const_iterator i_hist;
        cv::Mat_<uchar>::iterator i_labels;
        cv::Mat_<float>::iterator i_centres;
        uchar label;

        float sumDist = 0;
        int movedCount = 0;

        // Step 1. Assign each element a label
        for (i_bin = 0, i_labels = labels.begin(), i_hist = hist.begin();
             i_bin < nbins;
             ++i_bin, ++i_labels, ++i_hist)
        {
            float bin_val = binStart + i_bin*binWidth;
            float minDist = sq(bin_val - centres(*i_labels));
            int curLabel = *i_labels;

            for (label = 0; label < K; ++label)
            {
                float dist = sq(bin_val - centres(label));
                if (dist < minDist)
                {
                    minDist = dist;
                    *i_labels = label;
                }
            }

            if (*i_labels != curLabel)
                movedCount++;

            sumDist += (*i_hist) * std::sqrt(minDist);
        }

        if (finalRun)
            return sumDist;

        // Step 2. Recalculate centres
        cv::Mat_<float> counts(K, 1, 0.0f);
        for (i_bin = 0, i_labels = labels.begin(), i_hist = hist.begin();
             i_bin < nbins;
             ++i_bin, ++i_labels, ++i_hist)
        {
            float bin_val = binStart + i_bin*binWidth;

            centres(*i_labels) += (*i_hist) * bin_val;
            counts(*i_labels) += *i_hist;
        }
        for (label = 0; label < K; ++label)
        {
            if (counts(label) == 0)
                return std::numeric_limits<float>::infinity();

            centres(label) /= counts(label);
        }

        // Step 3. Detect termination criteria
        if (movedCount == 0)
            finalRun = true;
        else if (termCriteria.type | cv::TermCriteria::COUNT && iters >= termCriteria.maxCount)
            finalRun = true;
        else if (termCriteria.type | cv::TermCriteria::EPS)
        {
            float max_movement = 0;
            for (label = 0; label < K; ++label)
            {
                max_movement = std::max(max_movement, sq(centres(label) - old_centres(label)));
            }
            if (sqrt(max_movement) < termCriteria.epsilon)
                finalRun = true;
        }
    }
    return std::numeric_limits<float>::infinity();
}

cv::RotatedRect pupiltracker::cvx::fitEllipse(const cv::Moments& m)
{
    cv::RotatedRect ret;

    ret.center.x = m.m10/m.m00;
    ret.center.y = m.m01/m.m00;

    double mu20 = m.m20/m.m00 - ret.center.x*ret.center.x;
    double mu02 = m.m02/m.m00 - ret.center.y*ret.center.y;
    double mu11 = m.m11/m.m00 - ret.center.x*ret.center.y;

    double common = std::sqrt(sq(mu20 - mu02) + 4*sq(mu11));

    ret.size.width = std::sqrt(2*(mu20 + mu02 + common));
    ret.size.height = std::sqrt(2*(mu20 + mu02 - common));

    double num, den;
    if (mu02 > mu20) {
        num = mu02 - mu20 + common;
        den = 2*mu11;
    }
    else {
        num = 2*mu11;
        den = mu20 - mu02 + common;
    }

    if (num == 0 && den == 0)
        ret.angle = 0;
    else
        ret.angle = (180/PI) * std::atan2(num,den);

    return ret;
}
cv::Vec2f pupiltracker::cvx::majorAxis(const cv::RotatedRect& ellipse)
{
    return cv::Vec2f(ellipse.size.width*std::cos(PI/180*ellipse.angle), ellipse.size.width*std::sin(PI/180*ellipse.angle));
}

// Ellipse algorithm

namespace
{
    struct section_guard
    {
        std::string name;
        pupiltracker::tracker_log& log;
        pupiltracker::timer t;
        section_guard(const std::string& name, pupiltracker::tracker_log& log) : name(name), log(log), t() {  }
        ~section_guard() { log.add(name, t); }
        operator bool() const {return false;}
    };

    inline section_guard make_section_guard(const std::string& name, pupiltracker::tracker_log& log)
    {
        return section_guard(name,log);
    }
}

#define SECTION(A,B) if (const section_guard& _section_guard_ = make_section_guard( A , B )) {} else




class HaarSurroundFeature
{
public:
    HaarSurroundFeature(int r1, int r2) : r_inner(r1), r_outer(r2)
    {
        //  _________________
        // |        -ve      |
        // |     _______     |
        // |    |   +ve |    |
        // |    |   .   |    |
        // |    |_______|    |
        // |         <r1>    |
        // |_________<--r2-->|

        // Number of pixels in each part of the kernel
        int count_inner = r_inner*r_inner;
        int count_outer = r_outer*r_outer - r_inner*r_inner;

        // Frobenius normalized values
        //
        // Want norm = 1 where norm = sqrt(sum(pixelvals^2)), so:
        //  sqrt(count_inner*val_inner^2 + count_outer*val_outer^2) = 1
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Solving both of these gives:
        //val_inner = std::sqrt( (double)count_outer/(count_inner*count_outer + sq(count_inner)) );
        //val_outer = -std::sqrt( (double)count_inner/(count_inner*count_outer + sq(count_outer)) );

        // Square radius normalised values
        //
        // Want the response to be scale-invariant, so scale it by the number of pixels inside it:
        //  val_inner = 1/count = 1/r_outer^2
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Hence:
        val_inner = 1.0 / (r_inner*r_inner);
        val_outer = -val_inner*count_inner/count_outer;

    }

    double val_inner, val_outer;
    int r_inner, r_outer;
};

cv::RotatedRect fitEllipse(const std::vector<pupiltracker::EdgePoint>& edgePoints)
{
    std::vector<cv::Point2f> points;
    points.reserve(edgePoints.size());

    BOOST_FOREACH(const pupiltracker::EdgePoint& e, edgePoints)
        points.push_back(e.point);

    return cv::fitEllipse(points);
}


bool pupiltracker::findPupilEllipse(
    const pupiltracker::TrackerParams& params,
    const cv::Mat& m,

    pupiltracker::findPupilEllipse_out& out,
    pupiltracker::tracker_log& log
    )
{
    // --------------------
    // Convert to greyscale
    // --------------------

    cv::Mat_<uchar> mEye;

    SECTION("Grey and crop", log)
    {
        // Pick one channel if necessary, and crop it to get rid of borders
        if (m.channels() == 1)
        {
            mEye = m;
        }
        else if (m.channels() == 3)
        {
            cv::cvtColor(m, mEye, cv::COLOR_BGR2GRAY);
        }
        else if (m.channels() == 4)
        {
            cv::cvtColor(m, mEye, cv::COLOR_BGRA2GRAY);
        }
        else
        {
            throw std::runtime_error("Unsupported number of channels");
        }
    }

    // -----------------------
    // Find best haar response
    // -----------------------

    //             _____________________
    //            |         Haar kernel |
    //            |                     |
    //  __________|______________       |
    // | Image    |      |       |      |
    // |    ______|______|___.-r-|--2r--|
    // |   |      |      |___|___|      |
    // |   |      |          |   |      |
    // |   |      |          |   |      |
    // |   |      |__________|___|______|
    // |   |    Search       |   |
    // |   |    region       |   |
    // |   |                 |   |
    // |   |_________________|   |
    // |                         |
    // |_________________________|
    //

    cv::Mat_<int32_t> mEyeIntegral;
    int padding = 2*params.Radius_Max;

    SECTION("Integral image", log)
    {
        cv::Mat mEyePad;
        // Need to pad by an additional 1 to get bottom & right edges.
        cv::copyMakeBorder(mEye, mEyePad, padding, padding, padding, padding, cv::BORDER_REPLICATE);
        cv::integral(mEyePad, mEyeIntegral);
    }

    cv::Point2f pHaarPupil;
    int haarRadius = 0;

    SECTION("Haar responses", log)
    {
        const int rstep = 2;
        const int ystep = 4;
        const int xstep = 4;

        double minResponse = std::numeric_limits<double>::infinity();

        for (int r = params.Radius_Min; r < params.Radius_Max; r+=rstep)
        {
            // Get Haar feature
            int r_inner = r;
            int r_outer = 3*r;
            HaarSurroundFeature f(r_inner, r_outer);

            // Use TBB for rows
            std::pair<double,cv::Point2f> minRadiusResponse = tbb::parallel_reduce(
                tbb::blocked_range<int>(0, (mEye.rows-r - r - 1)/ystep + 1, ((mEye.rows-r - r - 1)/ystep + 1) / 8),
                std::make_pair(std::numeric_limits<double>::infinity(), UNKNOWN_POSITION),
                [&] (tbb::blocked_range<int> range, const std::pair<double,cv::Point2f>& minValIn) -> std::pair<double,cv::Point2f>
            {
                std::pair<double,cv::Point2f> minValOut = minValIn;
                for (int i = range.begin(), y = r + range.begin()*ystep; i < range.end(); i++, y += ystep)
                {
                    //            Š         Š
                    // row1_outer.|         |  p00._____________________.p01
                    //            |         |     |         Haar kernel |
                    //            |         |     |                     |
                    // row1_inner.|         |     |   p00._______.p01   |
                    //            |-padding-|     |      |       |      |
                    //            |         |     |      | (x,y) |      |
                    // row2_inner.|         |     |      |_______|      |
                    //            |         |     |   p10'       'p11   |
                    //            |         |     |                     |
                    // row2_outer.|         |     |_____________________|
                    //            |         |  p10'                     'p11
                    //            Š         Š

                    int* row1_inner = mEyeIntegral[y+padding - r_inner];
                    int* row2_inner = mEyeIntegral[y+padding + r_inner + 1];
                    int* row1_outer = mEyeIntegral[y+padding - r_outer];
                    int* row2_outer = mEyeIntegral[y+padding + r_outer + 1];

                    int* p00_inner = row1_inner + r + padding - r_inner;
                    int* p01_inner = row1_inner + r + padding + r_inner + 1;
                    int* p10_inner = row2_inner + r + padding - r_inner;
                    int* p11_inner = row2_inner + r + padding + r_inner + 1;

                    int* p00_outer = row1_outer + r + padding - r_outer;
                    int* p01_outer = row1_outer + r + padding + r_outer + 1;
                    int* p10_outer = row2_outer + r + padding - r_outer;
                    int* p11_outer = row2_outer + r + padding + r_outer + 1;

                    for (int x = r; x < mEye.cols - r; x+=xstep)
                    {
                        int sumInner = *p00_inner + *p11_inner - *p01_inner - *p10_inner;
                        int sumOuter = *p00_outer + *p11_outer - *p01_outer - *p10_outer - sumInner;

                        double response = f.val_inner * sumInner + f.val_outer * sumOuter;

                        if (response < minValOut.first)
                        {
                            minValOut.first = response;
                            minValOut.second = cv::Point(x,y);
                        }

                        p00_inner += xstep;
                        p01_inner += xstep;
                        p10_inner += xstep;
                        p11_inner += xstep;

                        p00_outer += xstep;
                        p01_outer += xstep;
                        p10_outer += xstep;
                        p11_outer += xstep;
                    }
                }
                return minValOut;
            },
                [] (const std::pair<double,cv::Point2f>& x, const std::pair<double,cv::Point2f>& y) -> std::pair<double,cv::Point2f>
            {
                if (x.first < y.first)
                    return x;
                else
                    return y;
            }
            );

            if (minRadiusResponse.first < minResponse)
            {
                minResponse = minRadiusResponse.first;
                // Set return values
                pHaarPupil = minRadiusResponse.second;
                haarRadius = r;
            }
        }
    }
    // Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
    haarRadius = (int)(haarRadius * SQRT_2);

    // ---------------------------
    // Pupil ROI around Haar point
    // ---------------------------
    cv::Rect roiHaarPupil = cvx::roiAround(cv::Point(pHaarPupil.x, pHaarPupil.y), haarRadius);
    cv::Mat_<uchar> mHaarPupil;
    cvx::getROI(mEye, mHaarPupil, roiHaarPupil);

    out.roiHaarPupil = roiHaarPupil;
    out.mHaarPupil = mHaarPupil;

    // --------------------------------------------------
    // Get histogram of pupil region, segment with KMeans
    // --------------------------------------------------

    const int bins = 256;

    cv::Mat_<float> hist;
    SECTION("Histogram", log)
    {
        int channels[] = {0};
        int sizes[] = {bins};
        float range[2] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&mHaarPupil, 1, channels, cv::Mat(), hist, 1, sizes, ranges);
    }

    out.histPupil = hist;

    float threshold;
    SECTION("KMeans", log)
    {
        // Try various candidate centres, return the one with minimal label distance
        float candidate0[2] = {0, 0};
        float candidate1[2] = {128, 255};
        float bestDist = std::numeric_limits<float>::infinity();
        float bestThreshold = std::numeric_limits<float>::quiet_NaN();

        for (int i = 0; i < 2; i++)
        {
            cv::Mat_<uchar> labels;
            float centres[2] = {candidate0[i], candidate1[i]};
            float dist = cvx::histKmeans(hist, 0, 256, 2, centres, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0.0));

            float thisthreshold = (centres[0] + centres[1])/2;
            if (dist < bestDist && boost::math::isnormal(thisthreshold))
            {
                bestDist = dist;
                bestThreshold = thisthreshold;
            }
        }

        if (!boost::math::isnormal(bestThreshold))
        {
            // If kmeans gives a degenerate solution, exit early
            return false;
        }

        threshold = bestThreshold;
    }

    cv::Mat_<uchar> mPupilThresh;
    SECTION("Threshold", log)
    {
        cv::threshold(mHaarPupil, mPupilThresh, threshold, 255, cv::THRESH_BINARY_INV);
    }

    out.threshold = threshold;
    out.mPupilThresh = mPupilThresh;

    // ---------------------------------------------
    // Find best region in the segmented pupil image
    // ---------------------------------------------

    cv::Rect bbPupilThresh;
    cv::RotatedRect elPupilThresh;

    SECTION("Find best region", log)
    {
        cv::Mat_<uchar> mPupilContours = mPupilThresh.clone();
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mPupilContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        if (contours.size() == 0)
            return false;

        std::vector<cv::Point>& maxContour = contours[0];
        double maxContourArea = cv::contourArea(maxContour);
        BOOST_FOREACH(std::vector<cv::Point>& c, contours)
        {
            double area = cv::contourArea(c);
            if (area > maxContourArea)
            {
                maxContourArea = area;
                maxContour = c;
            }
        }

        cv::Moments momentsPupilThresh = cv::moments(maxContour);

        bbPupilThresh = cv::boundingRect(maxContour);
        elPupilThresh = cvx::fitEllipse(momentsPupilThresh);

        // Shift best region into eye coords (instead of pupil region coords), and get ROI
        bbPupilThresh.x += roiHaarPupil.x;
        bbPupilThresh.y += roiHaarPupil.y;
        elPupilThresh.center.x += roiHaarPupil.x;
        elPupilThresh.center.y += roiHaarPupil.y;
    }

    out.bbPupilThresh = bbPupilThresh;
    out.elPupilThresh = elPupilThresh;

    // ------------------------------
    // Find edges in new pupil region
    // ------------------------------

    cv::Mat_<uchar> mPupil, mPupilOpened, mPupilBlurred, mPupilEdges;
    cv::Mat_<float> mPupilSobelX, mPupilSobelY;
    cv::Rect bbPupil;
    cv::Rect roiPupil = cvx::roiAround(cv::Point(elPupilThresh.center.x, elPupilThresh.center.y), haarRadius);
    SECTION("Pupil preprocessing", log)
    {
        const int padding = 3;

        cv::Rect roiPadded(roiPupil.x-padding, roiPupil.y-padding, roiPupil.width+2*padding, roiPupil.height+2*padding);
        // First get an ROI around the approximate pupil location
        cvx::getROI(mEye, mPupil, roiPadded, cv::BORDER_REPLICATE);

        cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mPupil, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2);

        if (params.CannyBlur > 0)
        {
            cv::GaussianBlur(mPupilOpened, mPupilBlurred, cv::Size(), params.CannyBlur);
        }
        else
        {
            mPupilBlurred = mPupilOpened;
        }

        cv::Sobel(mPupilBlurred, mPupilSobelX, CV_32F, 1, 0, 3);
        cv::Sobel(mPupilBlurred, mPupilSobelY, CV_32F, 0, 1, 3);

        cv::Canny(mPupilBlurred, mPupilEdges, params.CannyThreshold1, params.CannyThreshold2);

        cv::Rect roiUnpadded(padding,padding,roiPupil.width,roiPupil.height);
        mPupil = cv::Mat(mPupil, roiUnpadded);
        mPupilOpened = cv::Mat(mPupilOpened, roiUnpadded);
        mPupilBlurred = cv::Mat(mPupilBlurred, roiUnpadded);
        mPupilSobelX = cv::Mat(mPupilSobelX, roiUnpadded);
        mPupilSobelY = cv::Mat(mPupilSobelY, roiUnpadded);
        mPupilEdges = cv::Mat(mPupilEdges, roiUnpadded);

        bbPupil = cvx::boundingBox(mPupil);
    }

    out.roiPupil = roiPupil;
    out.mPupil = mPupil;
    out.mPupilOpened = mPupilOpened;
    out.mPupilBlurred = mPupilBlurred;
    out.mPupilSobelX = mPupilSobelX;
    out.mPupilSobelY = mPupilSobelY;
    out.mPupilEdges = mPupilEdges;

    // -----------------------------------------------
    // Get points on edges, optionally using starburst
    // -----------------------------------------------

    std::vector<cv::Point2f> edgePoints;

    if (params.StarburstPoints > 0)
    {
        SECTION("Starburst", log)
        {
            // Starburst from initial pupil approximation, stopping when an edge is hit.
            // Collect all edge points into a vector

            // The initial pupil approximations are:
            //    Centre of mass of thresholded region
            //    Halfway along the major axis (calculated form second moments) in each direction

            tbb::concurrent_vector<cv::Point2f> edgePointsConcurrent;

            cv::Vec2f elPupil_majorAxis = cvx::majorAxis(elPupilThresh);
            std::vector<cv::Point2f> centres;
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y));
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) + cv::Point2f(elPupil_majorAxis));
            centres.push_back(elPupilThresh.center - cv::Point2f(roiPupil.tl().x, roiPupil.tl().y) - cv::Point2f(elPupil_majorAxis));

            BOOST_FOREACH(const cv::Point2f& centre, centres) {
                tbb::parallel_for(0, params.StarburstPoints, [&] (int i) {
                    double theta = i * 2*PI/params.StarburstPoints;

                    // Initialise centre and direction vector
                    cv::Point2f pDir((float)std::cos(theta), (float)std::sin(theta));

                    int t = 1;
                    cv::Point p = centre + (t * pDir);
                    while(p.inside(bbPupil))
                    {
                        uchar val = mPupilEdges(p);

                        if (val > 0)
                        {
                            float dx = mPupilSobelX(p);
                            float dy = mPupilSobelY(p);

                            float cdirx = p.x - (elPupilThresh.center.x - roiPupil.x);
                            float cdiry = p.y - (elPupilThresh.center.y - roiPupil.y);

                            // Check edge direction
                            double dirCheck = dx*cdirx + dy*cdiry;

                            if (dirCheck > 0)
                            {
                                // We've hit an edge
                                edgePointsConcurrent.push_back(cv::Point2f(p.x + 0.5f, p.y + 0.5f));
                                break;
                            }
                        }

                        ++t;
                        p = centre + (t * pDir);
                    }
                });
            }

            edgePoints = std::vector<cv::Point2f>(edgePointsConcurrent.begin(), edgePointsConcurrent.end());


            // Remove duplicate edge points
            std::sort(edgePoints.begin(), edgePoints.end(), [] (const cv::Point2f& p1, const cv::Point2f& p2) -> bool {
                if (p1.x == p2.x)
                    return p1.y < p2.y;
                else
                    return p1.x < p2.x;
            });
            edgePoints.erase( std::unique( edgePoints.begin(), edgePoints.end() ), edgePoints.end() );

            if (edgePoints.size() < params.StarburstPoints/2)
                return false;
        }
    }
    else
    {
        SECTION("Non-zero value finder", log)
        {
            for(int y = 0; y < mPupilEdges.rows; y++)
            {
                uchar* val = mPupilEdges[y];
                for(int x = 0; x < mPupilEdges.cols; x++, val++)
                {
                    if(*val == 0)
                        continue;

                    edgePoints.push_back(cv::Point2f(x + 0.5f, y + 0.5f));
                }
            }
        }
    }


    // ---------------------------
    // Fit an ellipse to the edges
    // ---------------------------

    cv::RotatedRect elPupil;
    std::vector<cv::Point2f> inliers;
    SECTION("Ellipse fitting", log)
    {
        // Desired probability that only inliers are selected
        const double p = 0.999;
        // Probability that a point is an inlier
        double w = params.PercentageInliers/100.0;
        // Number of points needed for a model
        const int n = 5;

        if (params.PercentageInliers == 0)
            return false;

        if (edgePoints.size() >= n) // Minimum points for ellipse
        {
            // RANSAC!!!

            double wToN = std::pow(w,n);
            int k = static_cast<int>(std::log(1-p)/std::log(1 - wToN)  + 2*std::sqrt(1 - wToN)/wToN);

            out.ransacIterations = k;

            log.add("k", k);

            //size_t threshold_inlierCount = std::max<size_t>(n, static_cast<size_t>(out.edgePoints.size() * 0.7));

            // Use TBB for RANSAC
            struct EllipseRansac_out {
                std::vector<cv::Point2f> bestInliers;
                cv::RotatedRect bestEllipse;
                double bestEllipseGoodness;
                int earlyRejections;
                bool earlyTermination;

                EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
            };
            struct EllipseRansac {
                const TrackerParams& params;
                const std::vector<cv::Point2f>& edgePoints;
                int n;
                const cv::Rect& bb;
                const cv::Mat_<float>& mDX;
                const cv::Mat_<float>& mDY;
                int earlyRejections;
                bool earlyTermination;

                EllipseRansac_out out;

                EllipseRansac(
                    const TrackerParams& params,
                    const std::vector<cv::Point2f>& edgePoints,
                    int n,
                    const cv::Rect& bb,
                    const cv::Mat_<float>& mDX,
                    const cv::Mat_<float>& mDY)
                    : params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0)
                {
                }

                EllipseRansac(EllipseRansac& other, tbb::split)
                    : params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
                {
                    //std::cout << "Ransac split" << std::endl;
                }

                void operator()(const tbb::blocked_range<size_t>& r)
                {
                    if (out.earlyTermination)
                        return;
                    //std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
                    for( size_t i=r.begin(); i!=r.end(); ++i )
                    {
                        // Ransac Iteration
                        // ----------------
                        std::vector<cv::Point2f> sample;
                        if (params.Seed >= 0)
                            sample = randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
                        else
                            sample = randomSubset(edgePoints, n);

                        cv::RotatedRect ellipseSampleFit = fitEllipse(sample);
                        // Normalise ellipse to have width as the major axis.
                        if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
                        {
                            ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
                            std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
                        }

                        cv::Size s = ellipseSampleFit.size;
                        // Discard useless ellipses early
                        if (!ellipseSampleFit.center.inside(bb)
                            || s.height > params.Radius_Max*2
                            || s.width > params.Radius_Max*2
                            || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                            || s.height > 4*s.width
                            || s.width > 4*s.height
                            )
                        {
                            // Bad ellipse! Go to your room!
                            continue;
                        }

                        // Use conic section's algebraic distance as an error measure
                        ConicSection conicSampleFit(ellipseSampleFit);

                        // Check if sample's gradients are correctly oriented
                        if (params.EarlyRejection)
                        {
                            bool gradientCorrect = true;
                            BOOST_FOREACH(const cv::Point2f& p, sample)
                            {
                                cv::Point2f grad = conicSampleFit.algebraicGradientDir(p);
                                float dx = mDX(cv::Point(p.x, p.y));
                                float dy = mDY(cv::Point(p.x, p.y));

                                float dotProd = dx*grad.x + dy*grad.y;

                                gradientCorrect &= dotProd > 0;
                            }
                            if (!gradientCorrect)
                            {
                                out.earlyRejections++;
                                continue;
                            }
                        }

                        // Assume that the sample is the only inliers

                        cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
                        ConicSection conicInlierFit = conicSampleFit;
                        std::vector<cv::Point2f> inliers, prevInliers;

                        // Iteratively find inliers, and re-fit the ellipse
                        for (int i = 0; i < params.InlierIterations; ++i)
                        {
                            // Get error scale for 1px out on the minor axis
                            cv::Point2f minorAxis(-std::sin(PI/180.0*ellipseInlierFit.angle), std::cos(PI/180.0*ellipseInlierFit.angle));
                            cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
                            float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
                            float errorScale = 1.0f/errOf1px;

                            // Find inliers
                            inliers.reserve(edgePoints.size());
                            const float MAX_ERR = 2;
                            BOOST_FOREACH(const cv::Point2f& p, edgePoints)
                            {
                                float err = errorScale*conicInlierFit.distance(p);

                                if (err*err < MAX_ERR*MAX_ERR)
                                    inliers.push_back(p);
                            }

                            if (inliers.size() < n) {
                                inliers.clear();
                                continue;
                            }

                            // Refit ellipse to inliers
                            ellipseInlierFit = fitEllipse(inliers);
                            conicInlierFit = ConicSection(ellipseInlierFit);

                            // Normalise ellipse to have width as the major axis.
                            if (ellipseInlierFit.size.height > ellipseInlierFit.size.width)
                            {
                                ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
                                std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
                            }
                        }
                        if (inliers.empty())
                            continue;

                        // Discard useless ellipses again
                        s = ellipseInlierFit.size;
                        if (!ellipseInlierFit.center.inside(bb)
                            || s.height > params.Radius_Max*2
                            || s.width > params.Radius_Max*2
                            || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                            || s.height > 4*s.width
                            || s.width > 4*s.height
                            )
                        {
                            // Bad ellipse! Go to your room!
                            continue;
                        }

                        // Calculate ellipse goodness
                        double ellipseGoodness = 0;
                        if (params.ImageAwareSupport)
                        {
                            BOOST_FOREACH(cv::Point2f& p, inliers)
                            {
                                cv::Point2f grad = conicInlierFit.algebraicGradientDir(p);
                                float dx = mDX(p);
                                float dy = mDY(p);

                                double edgeStrength = dx*grad.x + dy*grad.y;

                                ellipseGoodness += edgeStrength;
                            }
                        }
                        else
                        {
                            ellipseGoodness = inliers.size();
                        }

                        if (ellipseGoodness > out.bestEllipseGoodness)
                        {
                            std::swap(out.bestEllipseGoodness, ellipseGoodness);
                            std::swap(out.bestInliers, inliers);
                            std::swap(out.bestEllipse, ellipseInlierFit);

                            // Early termination, if 90% of points match
                            if (params.EarlyTerminationPercentage > 0 && out.bestInliers.size() > params.EarlyTerminationPercentage*edgePoints.size()/100)
                            {
                                earlyTermination = true;
                                break;
                            }
                        }

                    }
                    //std::cout << "Ransac end" << std::endl;
                }

                void join(EllipseRansac& other)
                {
                    //std::cout << "Ransac join" << std::endl;
                    if (other.out.bestEllipseGoodness > out.bestEllipseGoodness)
                    {
                        std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
                        std::swap(out.bestInliers, other.out.bestInliers);
                        std::swap(out.bestEllipse, other.out.bestEllipse);
                    }
                    out.earlyRejections += other.out.earlyRejections;
                    earlyTermination |= other.earlyTermination;

                    out.earlyTermination = earlyTermination;
                }
            };

            EllipseRansac ransac(params, edgePoints, n, bbPupil, out.mPupilSobelX, out.mPupilSobelY);
            try
            {
                tbb::parallel_reduce(tbb::blocked_range<size_t>(0,k,k/8), ransac);
            }
            catch (std::exception& e)
            {
                const char* c = e.what();
                std::cerr << e.what() << std::endl;
            }
            inliers = ransac.out.bestInliers;
            log.add("goodness", ransac.out.bestEllipseGoodness);

            out.earlyRejections = ransac.out.earlyRejections;
            out.earlyTermination = ransac.out.earlyTermination;


            cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
            ConicSection conicBestFit(ellipseBestFit);
            BOOST_FOREACH(const cv::Point2f& p, edgePoints)
            {
                cv::Point2f grad = conicBestFit.algebraicGradientDir(p);
                float dx = out.mPupilSobelX(p);
                float dy = out.mPupilSobelY(p);

                out.edgePoints.push_back(EdgePoint(p, dx*grad.x + dy*grad.y));
            }

            elPupil = ellipseBestFit;
            elPupil.center.x += roiPupil.x;
            elPupil.center.y += roiPupil.y;
        }

        if (inliers.size() == 0)
            return false;

        cv::Point2f pPupil = elPupil.center;

        out.pPupil = pPupil;
        out.elPupil = elPupil;
        out.inliers = inliers;

        return true;
    }

    return false;
}

//Parameters for ellipse fitting algorithm
pupiltracker::TrackerParams params;

bool isDrawing = false;
Point start, boxend;


// currentDateTime:
// Returns the current date and time
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H%M%S", &tstruct);

    return buf;
}

//Scale images for debug output
void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

// Initialize Trackbars
void min_radius_trackbar(int, void*){
	if (min_radius_slider==0){
		min_radius_slider=1;
	}
	min_radius = (int) min_radius_slider;
}

void max_radius_trackbar(int,void*){
	max_radius = (int) max_radius_slider;
}

void canny_threshold1_trackbar(int,void*){
	canny_threshold1 = (int) canny_threshold1_slider;
}

void canny_threshold2_trackbar(int,void*){
	canny_threshold2 = (int) canny_threshold2_slider;
}

void canny_blur_trackbar(int,void*){
	canny_blur = (int) canny_blur_slider;
}

void starburst_pt_trackbar(int,void*){
	starburst_pt = (int) starburst_pt_slider;
}

void run_program_trackbar(int,void*){
	run_program = (int) run_program_slider;
}

void rec_trackbar(int,void*){
	record_video = (int) rec_slider;
}

void video_display_trackbar(int,void*){
	video_display = (int) video_display_slider;
}

void save_csv_trackbar(int,void*){
	save_csv = (int) save_csv_slider;
}

void stream_data_trackbar(int,void*){
	stream_data = (int) stream_data_slider;
}

void orientation_trackbar(int,void*){
	orientation = (int) orientation_slider;
}

void centerx_trackbar(int,void*){
	center_offset_x = xpos - cvFloor(0.5*max_rngx);
}

void centery_trackbar(int,void*){
	center_offset_y = ypos - cvFloor(0.5*max_rngy);
}

void offsetx_trackbar(int,void*){
	offsetx = (int) offsetx_slider - cvFloor(offsetx_slider_max*0.5);
}

void offsety_trackbar(int,void*){
	offsety = (int) offsety_slider - cvFloor(offsety_slider_max*0.5);
}

void downsample_trackbar(int, void*){
	downsample = (int) downsample_slider;
	/*
	if (downsample==0){
		downsample=1;
		downsample_slider=1;
	}
	*/
}

// Main function:
// This program will track the eye
int main(){
	// Setup comedi
        comedi_cmd cmdx;
	comedi_cmd cmdy;
        int err;
        int n,m, i;
        int total=0, n_chan = 0, freq = 80000;
        int subdevicex = -1;
	int subdevicey = -1;
        int verbose = 0;
        unsigned int chanlistx[2];
	unsigned int chanlisty[1];
        unsigned int maxdata_x;
        unsigned int maxdata_y;
        comedi_range *rng_x;
        comedi_range *rng_y;
        int ret;
        //struct parsed_options options;
        int fn;
        int aref = AREF_GROUND;
        int range = 0;
        int channelx = 0;
	int channely = 1;
        int buffer_length;
        subdevicex = -1;
	subdevicey = -1;

	n_chan = 2;

        devx = comedi_open(comdevice);
	devy = comedi_open(comdevice2);
        if(devx == NULL){
                fprintf(stderr, "error opening %s\n", comdevice);
                return -1;
        }

	if(devy == NULL){
		fprintf(stderr,"error opening %s\n", comdevice2);
		return -1;
	}

        if(subdevicex <0)
                subdevicex = comedi_find_subdevice_by_type(devx, COMEDI_SUBD_AO, 0);
        assert(subdevicex >= 0);

	if(subdevicey <0)
		subdevicey = comedi_find_subdevice_by_type(devy, COMEDI_SUBD_AO, 0);
	assert(subdevicey >= 0);



	maxdata_x = comedi_get_maxdata(devx, subdevicex, channelx);
        rng_x = comedi_get_range(devx, subdevicex, channelx, 0);
        max_rngx = maxdata_x;

	maxdata_y = comedi_get_maxdata(devy, subdevicey, channely);
        rng_y = comedi_get_range(devy, subdevicey, channely, 0);
	max_rngy = maxdata_y;

	// initialize timer rec
	double delay;
	CStopWatch sw;
<<<<<<< HEAD



=======
	
	  //buffers for heuristic filtering
  	boost::circular_buffer<double> buffer_x(4);
  	boost::circular_buffer<double> buffer_y(4);
  	double tmp1;
  	double tmp2;
	
	
>>>>>>> origin/master
	// save file
	cout << "\nChoose a file name to save to. Defaults to current date and time...\n";
	string input = "";
	string filename;
	string video_filename;
	getline(cin, input);
	if (input == ""){
		filename = currentDateTime();
		video_filename = currentDateTime();
	}
	else{
		filename = input;
		video_filename = input;
	}

	filename.append(".csv");
	const char *filen = filename.c_str();

	ofstream save_file (filen);

	// Initialize camera for setup
	FlyCapture2::Error error;
	Camera camera;
	CameraInfo camInfo;

	// Connect to the camera
	error = camera.Connect(0);
	if(error != PGRERROR_OK){
		std::cout << "failed to connect to camera..." << std::endl;
		return false;
	}

	error = camera.GetCameraInfo(&camInfo);
	if (error != PGRERROR_OK){
		std::cout << "failed to get camera info from camera" << std::endl;
		return false;
	}

	std::cout << camInfo.vendorName << " "
			<< camInfo.modelName << " "
			<< camInfo.serialNumber << std::endl;

	//testing modes
	//Format7PacketInfo fmt7PacketInfo;
	//Format7ImageSettings fmt7ImageSettings;
	//fmt7ImageSettings.width   = col_size;
	//fmt7ImageSettings.height  = row_size;
	//fmt7ImageSettings.mode    = MODE_8;
	//fmt7ImageSettings.offsetX = 312;
	//fmt7ImageSettings.offsetY = 0;
	//fmt7ImageSettings.pixelFormat = PIXEL_FORMAT_MONO8;
	//bool valid;
	//error = cam.ValidateFormat7Settings( &fmt7ImageSettings,
        //               &valid,
    	// &fmt7PacketInfo );
	//unsigned int num_bytes =
  	 // fmt7PacketInfo.recommendedBytesPerPacket;

	// Set Format 7 (partial image mode) settings
	//error = cam.SetFormat7Configuration( &fmt7ImageSettings,
        //                             num_bytes );
	//if ( error != PGRERROR_OK)
	//{
	//    PrintError( error );
	//    return -1;
	//}


	//stop testing


		error = camera.StartCapture();
	if(error==PGRERROR_ISOCH_BANDWIDTH_EXCEEDED){
		std::cout << "bandwidth exceeded" << std::endl;
		return false;
	}
	else if (error != PGRERROR_OK){
		std::cout << "failed to start image capture" << std::endl;
		return false;
	}




	// Setup: User positions eye in FOV
	//  Wait for 'c' to be pushed to move on
	cout << "Position eye inside field of view\n";
	cout << "ROI selection is now done automagically\n";
	cout << "press c to continue\n";
	char kb = 0;
	namedWindow("set",WINDOW_NORMAL);


	Image tmpImage;
	Image rgbTmp;
	cv::Mat tmp;
	while(kb != 'c'){
		// Grab frame from buffer
		FlyCapture2::Error error = camera.RetrieveBuffer(&tmpImage);
		if (error != PGRERROR_OK){
			std::cout<< "capture error" << std::endl;
			return false;
		}

		// Convert image to OpenCV color scheme
		tmpImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbTmp);

		unsigned int rowBytes = (double)rgbTmp.GetReceivedDataSize()/(double)rgbTmp.GetRows();

		tmp = cv::Mat(rgbTmp.GetRows(),rgbTmp.GetCols(),CV_8UC3,rgbTmp.GetData(),rowBytes);

		xmax = tmp.cols;
		ymax = tmp.rows;

		imshow("set",tmp);

		kb = cvWaitKey(30);
	}

	destroyWindow("set");


	// Initialize variables for sliders

	int dp = 1;

	// Min Dist
	min_dist = 1;

	// Max radius
	max_radius = 70;
	max_radius_slider = 70;
	max_radius_slider_max = 200;

	// Min radius
	min_radius = 30;
	min_radius_slider = 30;
	min_radius_slider_max = 199;

	// Canny Threshold 1
	canny_threshold1 = 20;
	canny_threshold1_slider_max = 100;
	canny_threshold1_slider = 20;

	// Canny Threshold 2
	canny_threshold2 = 40;
	canny_threshold2_slider_max = 100;
	canny_threshold2_slider = 40;

	// Canny blur
	canny_blur = 1;
	canny_blur_slider_max = 10;
	canny_blur_slider = 1;

	// No of Starburst Points
	starburst_pt = 30;
	starburst_pt_slider_max = 100;
	starburst_pt_slider = 30;

	// Video display
	video_display_slider_max = 1;
	video_display_slider = 1;
	video_display = 1;

	// Record video
	record_video = 0;
	rec_slider = 0;
	rec_slider_max = 1;

	orientation = 0;
	orientation_slider = 0;
	orientation_slider_max = 1;

	centerx_slider = 0;
	centery_slider = 0;
	// Set up window with ROI and offset
	Mat window;
	Rect myROI(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);
	offset[0] = coordinates[0];
	offset[1] = coordinates[1];

	// setup windows
	namedWindow("window",WINDOW_NORMAL);
	namedWindow("filtered",WINDOW_NORMAL);
	cvNamedWindow("control",WINDOW_NORMAL);
	resizeWindow("window",600,500);
	resizeWindow("filtered",250,200);
	resizeWindow("control",250,80);
	moveWindow("window",0,0);
	moveWindow("filtered",400,0);
	moveWindow("control",650,0);
	bool refresh = true;

	// Initialize video recorder
	VideoWriter vid;
	double fps = 20;
	Size S = Size((int) rgbTmp.GetCols(), (int) rgbTmp.GetRows());
	video_filename = video_filename.append("-video.avi");
	vid.open(video_filename,1196444237,fps,S,true);



	// make sliders
	createTrackbar("Min Radius", "control", &min_radius_slider,min_radius_slider_max, min_radius_trackbar);
	createTrackbar("Max Radius", "control", &max_radius_slider,max_radius_slider_max, max_radius_trackbar);
	createTrackbar("Canny Threshold 1", "control", &canny_threshold1_slider, canny_threshold1_slider_max, canny_threshold1_trackbar);
	createTrackbar("Canny Threshold 2", "control", &canny_threshold2_slider, canny_threshold2_slider_max, canny_threshold2_trackbar);
	createTrackbar("Canny blur","control",&canny_blur_slider,canny_blur_slider_max,canny_blur_trackbar);
	createTrackbar("No of Starburst Pts", "control", &starburst_pt_slider, starburst_pt_slider_max, starburst_pt_trackbar);
	createTrackbar("Record","control",&rec_slider,rec_slider_max,rec_trackbar);
	createTrackbar("Orientation","control",&orientation_slider,orientation_slider_max,orientation_trackbar);
	createTrackbar("center-x","control",&centerx_slider,1,centerx_trackbar);
	createTrackbar("center-y","control",&centery_slider,1,centery_trackbar);
	createTrackbar("downsample","control",&downsample_slider,downsample_slider_max,downsample_trackbar);
	sw.Start(); // start timer
	char key = 0;

	int reset = 1000;
	int iter = 0;


	// This is the main loop for the function
	while(key != 'q'){

		//start timer
		Image rawImage;
		FlyCapture2::Error error = camera.RetrieveBuffer( &rawImage );
		if (error != PGRERROR_OK ){
			std::cout << "capture error" << std::endl;
			continue;
		}

		Image rgbImage;
		rawImage.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage);
		// convert to OpenCV Mat
		unsigned int rowBytes = (double)rgbImage.GetReceivedDataSize()/(double)rgbImage.GetRows();
		Mat image = Mat(rgbImage.GetRows(), rgbImage.GetCols(),CV_8UC3, rgbImage.GetData(),rowBytes);

		params.Radius_Min = min_radius;
		params.Radius_Max = max_radius;
		params.CannyBlur = canny_blur;
		params.CannyThreshold1 = canny_threshold1;
		params.CannyThreshold2 = canny_threshold2;
		params.StarburstPoints = starburst_pt;
		params.PercentageInliers = 30;
		params.InlierIterations = 2;
		params.ImageAwareSupport = false;
		params.EarlyTerminationPercentage = 95;
		params.EarlyRejection = true;
		params.Seed = -1;

		pupiltracker::findPupilEllipse_out out;
        	pupiltracker::tracker_log log;
        	pupiltracker::findPupilEllipse(params, image, out, log);
 		pupiltracker::cvx::cross(image, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
        	cv::ellipse(image, out.elPupil, pupiltracker::cvx::rgb(255,0,255));

		//scaling to output
		//we also assume that the pupil cant be at the VERY edge of the FOV
<<<<<<< HEAD
		xpos = ((out.pPupil.x - 16) / (xmax-32))*(double)max_rngx;
		ypos = ((out.pPupil.y - 16) / (ymax-32))*(double)max_rngy;


		//downsampling

                n = SAMPLE_CT * sizeof(sampl_t);
                for (i=0; i<SAMPLE_CT; i++){
			dataxl[0] = xpos;
			datayl[0] = ypos;
                }
		std::cout << "\r" << dataxl[0] << "," << datayl[0] << std::flush;

=======
		xpos = ((out.pPupil.x - 100) / (xmax-200))*(double)max_rngx;
		ypos = ((out.pPupil.y - 100) / (ymax-200))*(double)max_rngy;
		
		if(buffer_x.size() < 3){
      buffer_x.push_front(xpos);
      buffer_y.push_front(ypos);
    }
    else{
    buffer_x.push_front(xpos);
    buffer_y.push_front(ypos);
    // filter level 1
    if(buffer_x[2] > buffer_x[1] && buffer_x[1] < buffer_x[0]){
      tmp1 = std::abs(buffer_x[1] - buffer_x[0]);
      tmp2 = std::abs(buffer_x[1] - buffer_x[2]);

      if(tmp2 > tmp1){
        buffer_x[1] = buffer_x[0];
      }
      else{
        buffer_x[1] = buffer_x[2] ;
      }
      buffer_x[3] = buffer_x[2];
      buffer_x[2] = buffer_x[1];
    }
    else if(buffer_x[2] < buffer_x[1] && buffer_x[1] > buffer_x[0]){
      tmp1 = std::abs(buffer_x[1] - buffer_x[0]);
      tmp2 = std::abs(buffer_x[1] - buffer_x[2]);

      if(tmp2 > tmp1){
        buffer_x[1] = buffer_x[0];
      }
      else{
        buffer_x[1] = buffer_x[2] ;
      }

      buffer_x[3] = buffer_x[2];
      buffer_x[2] = buffer_x[1];
    }
    else{
      buffer_x[3] = buffer_x[2];
      buffer_x[2] = buffer_x[1];
    }
    if(buffer_y[2] > buffer_y[1] && buffer_y[1] < buffer_y[0]){
      tmp1 = std::abs(buffer_y[1] - buffer_y[0]);
      tmp2 = std::abs(buffer_y[1] - buffer_y[2]);

      if(tmp2 > tmp1){
        buffer_y[1] = buffer_y[0];
      }
      else{
        buffer_y[1] = buffer_y[2] ;
      }
      buffer_y[3] = buffer_y[2];
      buffer_y[2] = buffer_y[1];
    }
    else if(buffer_y[2] < buffer_y[1] && buffer_y[1] > buffer_y[0]){
      tmp1 = std::abs(buffer_y[1] - buffer_y[0]);
      tmp2 = std::abs(buffer_y[1] - buffer_y[2]);

      if(tmp2 > tmp1){
        buffer_y[1] = buffer_y[0];
      }
      else{
        buffer_y[1] = buffer_y[2] ;
      }

      buffer_y[3] = buffer_y[2];
      buffer_y[2] = buffer_y[1];
    }
    else{
      buffer_y[3] = buffer_y[2];
      buffer_y[2] = buffer_y[1];
    }

    //downsampling

    //  n = SAMPLE_CT * sizeof(sampl_t);
      //for (i=0; i<SAMPLE_CT; i++){
			dataxl[0] = buffer_x[2];
			datayl[0] = buffer_y[2];
              //  }
		//std::cout << "\r" << dataxl[0] << "," << datayl[0] << std::flush;


		ret = comedi_internal_trigger_cust(devx,subdevicex,channelx, channely,dataxl,datayl,range,aref);

    }

		if (ret < 0){
			comedi_perror("insn error");
		}

		usleep(1.1e1);

		// Record the video - this is slow!!
		if (record_video == 1){
			vid.write(image);
			sw.Stop();
	                delay = sw.GetDuration();
		}


		if (video_display==1 or save_csv==1){
			if (video_display==1){
				imshow("window",image);
				imshow("filtered", out.mPupilEdges);
			sw.Stop();
	                delay = sw.GetDuration();

			//std::cout << "\r" << delay << std::flush;
			}
		}
		
>>>>>>> origin/master

		ret = comedi_internal_trigger_cust(devx,subdevicex,channelx, channely,dataxl,datayl,range,aref);

		if (ret < 0){
			comedi_perror("insn error");
		}

		usleep(1.1e1);

		// Record the video - this is slow!!
		if (record_video == 1){
			vid.write(image);
			sw.Stop();
	                delay = sw.GetDuration();
		}


		if (video_display==1 or save_csv==1){
			if (video_display==1){
				imshow("window",image);
				imshow("filtered", out.mPupilEdges);
			sw.Stop();
	                delay = sw.GetDuration();

			//std::cout << "\r" << delay << std::flush;
			}
		}

		key = waitKey(1);
		sw.Start(); // restart timer

	}

}
