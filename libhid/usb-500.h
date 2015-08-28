/*
 *
 *  Copyright (c) 2006  Warren Jasper <wjasper@tx.ncsu.edu>
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

#ifndef USB_500_H
#define USB_500_H

#ifdef __cplusplus
extern "C" {
#endif

#include <time.h>

//#define USB_DEBUG

#define USB500_MAX_VALUES (0x4000)   // maximum number of samples 16384

#define USB500_WAIT_READ  500
#define USB500_WAIT_WRITE 100

#define USB500_VID 0x10c4  // Cygnal Integrated Products (Silicon Laboratories)
#define USB500_PID 0x0002  // EL-USB-[1,2]  MCC USB-50[1,2]

#define HIGH_ALARM_STATE (0x1 << 0)     // 1 = High Alarm Enabled, 0 = High Alarm Disabled
#define LOW_ALARM_STATE  (0x1 << 1)     // 1 = Low Alarm Enabled,  0 = Low Alarm Disabled
#define HIGH_ALARM_LATCH (0x1 << 2)     // 1 = High Alarm Indication Hold Enabled
                                        // 0 = High Alarm Indication Hold Disabled
#define LOW_ALARM_LATCH  (0x1 << 3)     // 1 = Low Alarm Indication Hold Enabled                 EL-USB-2 Only
                                        // 0 = Low Alarm Indication Hold Disabled
#define CH2_HIGH_ALARM_STATE (0x1 << 4) // 1 = High Alarm Enabled, 0 = High Alarm Disabled       EL-USB-2 Only
#define CH2_LOW_ALARM_STATE  (0x1 << 5) // 1 = High Alarm Enabled, 0 = High Alarm Disabled       
#define CH2_HIGH_ALARM_LATCH (0x1 << 6) // 1 = High Alarm Indication Hold Enabled                EL-USB-2 Only
                                        // 0 = High Alarm Indication Hold Disabled               
#define CH2_LOW_ALARM_LATCH (0x1 << 7)  // 1 = Low Alarm Indication Hold Enabled                 EL-USB-2 Only
                                        // 0 = Low Alarm Indication Hold Disabled
#define LOGGING_STATE (0x1 << 8)        // 1 = Delayed Start or Logging,  0 = Off
#define UNREAD        (0x1 << 9)        // 1 = Data has NOT been downloaded.
                                        // 0 = Data has been downloaded.
#define BATTERY_LOW   (0x1 << 10)       // 1 = During the last acquistion battery
                                        //     level dropped to a low level 
                                        //     and logging stopped
#define BATTERY_FAIL  (0x1 << 11)       // 1 = During the last acquistion battery
                                        //     level dropped to a critical level 
                                        //     and logging stopped
#define SENSOR_FAIL   (0x1 << 12)       // 1 = During last acquisition there were sensor errors. EL-USB-2 Only

/* 
  Note:
    EL-USB clears after current data has been downloaded.
*/

typedef struct t_configurationBlock {
  __u8 type;               // 1-7
  __u8 command;            //
  char name[16];           // Null terminated
  __u8 startTimeHours;     // 0-23
  __u8 startTimeMinutes;   // 0-59
  __u8 startTimeSeconds;   // 0-59
  __u8 startTimeDay;       // 1-31
  __u8 startTimeMonth;     // 1-12
  __u8 startTimeYear;      // 00-99
  __u32 startTimeOffset;   // 0-4294967295
  __u16 sampleRate;        // 0-65535
  __u16 sampleCount;       // 0-16382
  __u16 flagBits;          //
  union {
    __u16 highAlarm;       // 0-65535   USB-503 & USB-504
    struct {
      __u8 highAlarm;      // 0-255 USB-501 & USB-502
      __u8 lowAlarm;       // 0-255
    } channel1;
  } alarm1;
  float calibrationMValue; // 3.4E +/- 38 (7dig)
  float calibrationCValue;         // 3.4E +/- 38 (7dig)
  __u16 rawInputReading;   // 0 - 65535 (little endian)
  __u8 inputType;          // 0-1
  __u8 flagBits2;          // USB-503 & USB-504 only.
  char version[4];         // Not Null terminated
  __u32 serialNumber;      // 0-4294967295
  union {
    __u16 lowAlarm;        // 0-65535   USB-503 & USB-504
    struct {
      __u8 highAlarm;      // 0-255 USB-501 & USB-502
      __u8 lowAlarm;       // 0-255
    } channel2;
  } alarm2;
  __u8 notUsed[6];
  char displayUnitText[12];
  char calibrationInput1Text[8];
  char calibrationOutput1Text[8];
  char calibrationInput2Text[8];
  char calibrationOutput2Text[8];
  float capScalingFactor;
  char highAlarmLevelText[8];
  char lowAlarmLevelText[8];
  char defaultRangeDescriptionText[14];
  char defaultInputUnitText[12];
  char defaultDisplayUnit[12];
  char defaultCalibrationInput1Text[8];
  char defaultCalibrationOutput1Text[8];
  char defaultCalibrationInput2Text[8];
  char defaultCalibrationOutput2Text[8];
  char defaultHighAlarmLevelText[8];
  char defaultLowAlarmLevelText[8];
  __u16 lowBatteryAlarm;
  char notUsed2[40];
} configurationBlock;

typedef struct t_usb501_data {
  time_t time;         // time sample was taken
  float  temperature;  // temperature
} usb501_data;

typedef struct t_usb502_data {
  time_t time;         // time sample was taken
  float  temperature;  // temperature
  float  humidity;     // humidity (USB-502 only)
} usb502_data;

typedef struct t_usb503_data {
  time_t time;         // time sample was taken
  float  voltage;      // 0-30V range
} usb503_data;

typedef struct t_usb504_data {
  time_t time;         // time sample was taken
  float  current;      // 4-20 mA
} usb504_data;

float celsius2fahr( float celsius );
float fahr2celsius( float fahr );
void cleanup_USB500( usb_dev_handle *udev );
usb_dev_handle* usb_device_find_USB500( int vendorId, int productId );
void unlock_device_USB500( usb_dev_handle *udev );
void lock_device_USB500(usb_dev_handle *udev);
int read_configuration_block_USB500( usb_dev_handle *udev, configurationBlock *cblock );
void write_configuration_block_USB500( usb_dev_handle *udev, configurationBlock *cblock );
void stop_logging_USB500( usb_dev_handle *udev, configurationBlock *cblock );
void start_logging_USB500( usb_dev_handle *udev, configurationBlock *cblock );
void set_alarm( usb_dev_handle *udev, configurationBlock *cblock );

int read_recorded_data_USB501( usb_dev_handle *udev, configurationBlock *cblock, usb501_data *data);
int read_recorded_data_USB502( usb_dev_handle *udev, configurationBlock *cblock, usb502_data *data);
int read_recorded_data_USB503( usb_dev_handle *udev, configurationBlock *cblock, usb503_data *data);
int read_recorded_data_USB504( usb_dev_handle *udev, configurationBlock *cblock, usb504_data *data);
 
void write_recorded_data_USB501( configurationBlock *cblock, usb501_data *data_501 );
void write_recorded_data_USB502( configurationBlock *cblock, usb502_data *data_502 );
void write_recorded_data_USB503( configurationBlock *cblock, usb503_data *data_503 );
void write_recorded_data_USB504( configurationBlock *cblock, usb504_data *data_504 );

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif //USB_500_H
