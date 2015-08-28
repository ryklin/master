#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <math.h>
#include <sys/types.h>
#include <asm/types.h>
#include <time.h>

#include "pmd.h"
#include "usb-20X.h"


#define MAX_COUNT     (0xffff)
#define FALSE 0
#define TRUE 1

//timespec diff(timespec start, timespec end);

int toContinue()
{
  int answer;
  answer = 0; //answer = getchar();
  printf("Continue [yY]? ");
  while((answer = getchar()) == '\0' ||
    answer == '\n');
  return ( answer == 'y' || answer == 'Y');
}

int main (int argc, char **argv){
  float DIODE_THRESHOLD = -.02;
  float XPOS_THRESHOLD = .1;
  FILE *f;
  FILE *raw;

  f = fopen("data.txt","a");
  raw = fopen("raw.csv","a");

  usb_dev_handle *udev = NULL;

  float table_AIN[NCHAN_USB20X][2];

  int i;
  __u16 l;
  __u16 p;
    float vl;
    float vp = -1;

  udev = NULL;
  if ((udev = usb_device_find_USB_MCC(USB201_PID))) {
    printf("Found a USB 201\n");
  } else {
    printf("Failure, did not find DAQ\n");
    return 0;
  }

  // some initialization

  //print out the wMaxPacketSize.  Should be 512
  printf("wMaxPacketSize = %d\n", usb_get_max_packet_size(udev,0));

  usbBuildGainTable_USB20X(udev, table_AIN);
  for (i = 0; i < NCHAN_USB20X; i++) {
    printf("Calibration Table: %d   Slope = %f   Offset = %f\n", i, table_AIN[i][0], table_AIN[i][1]);
  }

//  ofstream save_file ("latency.csv");


  int key = 0;
  char CH0 = '0';
  char CH1 = '1';
  while(!key) {
    l = usbAIn_USB20X(udev, CH0);
    vl = volts_USB20X(udev,l);
    p = usbAIn_USB20X(udev, CH1);
    vp = volts_USB20X(udev, p);
    fprintf(raw,"%f, %f, %f \n",vl,vp,(float)clock());

    if (vl < DIODE_THRESHOLD) {
      clock_t start = clock(), diff;
      while(vp < XPOS_THRESHOLD){
	l = usbAIn_USB20X(udev, CH0);
	vl = volts_USB20X(udev,l);
        p = usbAIn_USB20X(udev, CH1);
        vp = volts_USB20X(udev,p);
	fprintf(raw,"%f, %f, %f \n",vl,vp,(float)clock());
      }
      diff = clock() - start;
      float d = (float)diff * 1000 / CLOCKS_PER_SEC;
      printf("%f \n",d);
      fprintf(f,"%f, ",d);
      fclose(f);
      f = fopen("time.csv","a");
      while(vp > XPOS_THRESHOLD){
	l = usbAIn_USB20X(udev, CH0);
	vl = volts_USB20X(udev, l);
        p = usbAIn_USB20X(udev, CH1);
        vp = volts_USB20X(udev, p);
	fprintf(raw,"%f, %f, %f \n",vl,vp,(float)clock());
      }
      fclose(raw);
      raw = fopen("raw.csv","a");
    }
  }
  return 0;
}

