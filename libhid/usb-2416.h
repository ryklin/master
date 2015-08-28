/*
 *  Copyright (c) 2009-2013  Warren Jasper <wjasper@tx.ncsu.edu>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef USB_2416_H

#define USB_2416_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB2416_PID     (0x00d0)
#define USB2416_4AO_PID (0x00d1)

/* Description of the requestType byte */
// Data transfer direction D7
#define HOST_TO_DEVICE (0x0 << 7)
#define DEVICE_TO_HOST (0x1 << 7)
// Type D5-D6
#define STANDARD_TYPE (0x0 << 5)
#define CLASS_TYPE    (0x1 << 5)
#define VENDOR_TYPE   (0x2 << 5)
#define RESERVED_TYPE (0x3 << 5)
// Recipient D0 - D4
#define DEVICE_RECIPIENT    (0x0)
#define INTERFACE_RECIPIENT (0x1)
#define ENDPOINT_RECIPIENT  (0x2)
#define OTHER_RECIPIENT     (0x3)
#define RESERVED_RECIPIENT  (0x4) 

/* Commands and HID Report ID for USB 2416  */
/* Digital I/O Commands */
#define DIN              (0x00)     // Read digital port
#define DOUT             (0x01)     // Read/Write digital port drive register

/* Analog Input Commands */
#define AIN              (0x10)     // Read analog input channel
#define AIN_SCAN_START   (0x11)     // Start analog input scan
#define AIN_SCAN_STOP    (0x12)     // Stop analog input scan
#define AIN_SCAN_STATUS  (0x13)     // Read analog input scan status
#define AIN_SCAN_QUEUE   (0x14)     // Read/Write analog input channel gain queue
#define SETTLING_SCAN    (0x16)     // Front end settling test

/* Analog Output Commands */
#define AOUT             (0x18)     // Write analog output channel
#define AOUT_SCAN_START  (0x19)     // Analog output scan start
#define AOUT_SCAN_STOP   (0x1A)     // Analog output scan stop
#define AOUT_SCAN_STATUS (0x1B)     // Analog output scan status

/* Counter Commands */
#define COUNTER          (0x20)     // Read/reset event counter

/* Memory Commands */
#define MEMORY           (0x30)     // Read/Write EEPROM

/* Miscellaneous Commands */
#define RESET            (0x40)     // Reset the device
#define BLINK_LED        (0x41)     // Causes LED to blink
#define CJC              (0x42)     // Read CJC sensor values
#define CAL_CONFIG       (0x43)     // Configure calibration source
#define GET_STATUS       (0x44)     // Read device status
#define ADCAL            (0x45)     // Perform A/D self-calibration
#define TC_CAL           (0x46)     // Measure TC calibration source
#define SERIAL           (0x48)     // Read/Write USB Serial Number
#define VERSION          (0x49)     // Read micro firmware versions

//  Modes
#define DIFFERENTIAL     (0x0)      // Voltage - differential
#define SE_HIGH          (0x1)      // Voltage - single-ended high channel
#define SE_LOW           (0x2)      // Voltage - single-ended low channel
#define DAC_READBACK     (0x3)      // Voltage - D/A readback
#define THERMOCOUPLE     (0x4)      // Thermocouple
#define CAL_OFFSET       (0x5)      // AIn Offset Calibration
#define CAL_GAIN         (0x6)      // AIn Gain Calibration
#define TC_OFFSET        (0x7)      // TC Offset Calibration
#define TC_GAIN_POS      (0x8)      // TC Gain Calibration Positive
#define TC_GAIN_NEG      (0x9)      // TC Gain Calibration Negative
#define BURNOUT          (0xa)      // Thermocouple without burnout detect

// Gain Ranges
#define BP_20V           (0x0)     // +/- 20V
#define BP_10V           (0x1)     // +/- 10V
#define BP_5V            (0x2)     // +/- 5V
#define BP_2_5V          (0x3)     // +/- 2.5V
#define BP_1_25V         (0x4)     // +/- 1.25V
#define BP_625V          (0x5)     // +/- 0.625V
#define BP_312V          (0x6)     // +/- 0.312V
#define BP_156V          (0x7)     // +/- 0.156V
#define BP_078V          (0x8)     // +/- 0.078V Voltage (all), Thermocouple

// Rates
#define HZ30000          (0)       // 30,000 S/s
#define HZ15000          (1)       // 15,000 S/s
#define HZ7500           (2)       //  7,500 S/s
#define HZ3750           (3)       //  3,750 S/s
#define HZ2000           (4)       //  2,000 S/s
#define HZ1000           (5)       //  1,000 S/s
#define HZ500            (6)       //    500 S/s
#define HZ100            (7)       //    100 S/s
#define HZ60             (8)       //     60 S/s
#define HZ50             (9)       //     50 S/s
#define HZ30             (10)      //     30 S/s
#define HZ25             (11)      //     25 S/s
#define HZ15             (12)      //     15 S/s
#define HZ10             (13)      //     10 S/s
#define HZ5              (14)      //      5 S/s
#define HZ2_5            (15)      //    2.5 S/s

#define COUNTER0         0x0       //  Counter 0
#define COUNTER1         0x1       //  Counter 1
  
#define NCHAN_2416      32       // max number of A/D channels in the device
#define NGAINS_2416     10       // max number of gain levels (analog input)
#define MAX_QUEUE_SIZE  64       // max number of entries in the AIN scan queue
#define NCHAN_AO_2416    4       // number of analog output channels

/* Define the types of Thermocouples supported */
#define TYPE_J	0
#define TYPE_K	1
#define TYPE_T	2
#define TYPE_E	3
#define TYPE_R	4
#define TYPE_S	5
#define TYPE_B	6
#define TYPE_N	7

typedef struct t_AInScanQueue {
  __u8 count;
  struct t_queue {
    __u8 channel;
    __u8 mode;
    __u8 range;
    __u8 rate;
  } queue[MAX_QUEUE_SIZE];
} AInScanQueue;

#define INPUT_SCAN_RUNNING (0x1)
#define INPUT_FIFO_FULL (0x2)
#define INPUT_PACER_SHORT (0x4)

#define OUTPUT_SCAN_RUNNING (0x1)
#define OUTPUT_SCAN_UNDERRUN (0x2)

void cleanup_USB2416(usb_dev_handle *udev);
void usbBlink_USB2416(usb_dev_handle *udev,  __u8 count);
__u8 usbDIn_USB2416(usb_dev_handle *udev, __u8 port);
void usbDOut_USB2416(usb_dev_handle *udev, __u8 value, __u8 port);
__u8 usbDOutR_USB2416(usb_dev_handle *udev, __u8 port);
int  usbAIn_USB2416(usb_dev_handle *udev, __u8 channel, __u8 mode, __u8 range, __u8 rate, __u8 *flags);
void usbAInScanStop_USB2416(usb_dev_handle *udev);
__u8 usbAInScanStatus_USB2416(usb_dev_handle *udev, __u16 *depth);
void usbAInScanQueueWrite_USB2416(usb_dev_handle *udev, AInScanQueue *queue);
void usbAInScanQueueRead_USB2416(usb_dev_handle *udev, AInScanQueue *queue);
void usbAInScanStart_USB2416(usb_dev_handle *udev, double frequency, __u16 count, __u8 packet_size, int *data);
void usbGetSerialNumber_USB2416(usb_dev_handle *udev, char serial[9]);
void usbSetSerialNumber_USB2416(usb_dev_handle *udev, char serial[9]);
void usbGetVersion_USB2416(usb_dev_handle *udev, __u16 version[4]);
void usbReset_USB2416(usb_dev_handle *udev);
void usbCounterInit_USB2416(usb_dev_handle *udev, __u8 counter);
__u32 usbCounter_USB2416(usb_dev_handle *udev, __u8 counter);
__u8 usbStatus_USB2416(usb_dev_handle *udev);
void usbCJC_USB2416(usb_dev_handle *udev, float temp[8]);  
void usbReadMemory_USB2416(usb_dev_handle *udev, __u16 length,  __u16 address, __u8 *data);
void usbWriteMemory_USB2416(usb_dev_handle *udev, __u16 length,  __u16 address, __u8 *data);
void usbCalConfig_USB2416(usb_dev_handle *udev, __u8 value);
void usbADCal_USB2416(usb_dev_handle *udev);
void usbTCCalMeasure(usb_dev_handle *udev, __u8 value);
double volts_USB2416(usb_dev_handle *udev, const int gain, const int value);
void usbBuildGainTable_USB2416(usb_dev_handle *udev, double table[NGAINS_2416][2]);

/* USB 2416_4AO specific */
void usbBuildGainTable_USB2416_4AO(usb_dev_handle *udev, double table_AO[NCHAN_AO_2416][2]);
void usbAOutScanStop_USB2416_4AO(usb_dev_handle *udev);
void usbAOutScanStart_USB2416_4AO(usb_dev_handle *udev, double frequency, __u16 scans, __u8 options);
__u8  usbAOutScanStatus_USB2416_4AO(usb_dev_handle *udev, __u16 *depth);
void usbAOut_USB2416_4AO(usb_dev_handle *udev, int channel, double voltage, double table_A0[NCHAN_AO_2416][2]);
void voltsTos16_USB2416_4AO(double *voltage, __s16 *data, int nSamples, double table_AO[]);
double usbAInMinPacerPeriod_USB2416(usb_dev_handle *udev);
int  sint24TOint(int int24val);
double NISTCalcVoltage(unsigned char, double);
double NISTCalcTemp(unsigned char, double);
double tc_temperature_USB2416(usb_dev_handle *udev, int tc_type, __u8 channel, double table_AI[NGAINS_2416][2]);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif // USB_2416_H
