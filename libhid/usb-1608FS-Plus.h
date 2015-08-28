/*
 *
 *  Copyright (c) 2013  Warren Jasper <wjasper@tx.ncsu.edu>
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

#ifndef USB_1608FS_PLUS_H

#define USB_1608FS_PLUS_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB1608FS_PLUS_PID (0x00ea)

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

/* Commands and HID Report ID for USB 1608FS-Plus  */
/* Digital I/O Commands */
#define DTRISTATE     (0x00)   // Read/Write Tristate register
#define DPORT         (0x01)   // Read digital port pins
#define DLATCH        (0x02)   // Read/Write Digital port output latch register

/* Analog Input Commands */
#define AIN            (0x10)  // Read analog input channel
#define AIN_SCAN_START (0x11)  // Start analog input scan
#define AIN_SCAN_STOP  (0x12)  // Stop analog input scan
#define AIN_CONFIG     (0x14)  // Analog input channel configuration
#define AIN_CLR_FIFO   (0x15)  // Clear the analog input scan FIFO

/* Counter/Timer Commands */
#define COUNTER           (0x20)  // Read/reset event counter

/* Memory Commands */
#define CAL_MEMORY        (0x30)  // Read/Write Calibration Memory
#define USER_MEMORY       (0x31)  // Read/Write User Memory
#define MBD_MEMORY        (0x32)  // Read/Write MBD Memory

/* Miscellaneous Commands */  
#define BLINK_LED         (0x41)  // Causes LED to blink
#define RESET             (0x42)  // Reset device
#define STATUS            (0x44)  // Read device status
#define SERIAL            (0x48)  // Read/Write serial number
#define DFU               (0x50)  // Enter device firmware upgrade mode

/* MBD */
#define MBD_COMMAND       (0x80) // Text-based MBD command / response
#define MBD_RAW           (0x81) // Raw MBD response

/* Analog Input Scan Options */
#define IMMEDIATE_TRANSFER_MODE (0x1)
#define BLOCK_TRANSFER_MODE     (0x0)
#define INTERNAL_PACER_ON       (0x2) // output internal pacer on SYNC
#define INTERNAL_PACER_OFF      (0x0) 
#define NO_TRIGGER              (0x0)
#define TRIG_EDGE_RISING        (0x1 << 2)
#define TRIG_EDGE_FALLING       (0x2 << 2)
#define TRIG_LEVEL_HIGH         (0x3 << 2)
#define TRIG_LEVEL_LOW          (0x4 << 2)
#define DEBUG_MODE              (0x10)
#define STALL_ON_OVERRUN        (0x0)
#define INHIBIT_STALL           (0x1 << 7)

/* Aanalog Input */
#define SINGLE_ENDED   0
#define DIFFERENTIAL   1
#define CALIBRATION    3
#define LAST_CHANNEL   (0x80)
#define PACKET_SIZE    64       // max bulk transfer size in bytes
  
/* Ranges */
#define BP_10V   0x0      // +/- 10 V
#define BP_5V    0x1      // +/- 5V
#define BP_2_5V  0x2      // +/- 2.5V
#define BP_2V    0x3      // +/- 2V
#define BP_1_25V 0x4      // +/- 1.25V
#define BP_1V    0x5      // +/- 1V
#define BP_625V  0x6      // +/- 0.625V
#define BP_3125V 0x7      // +/- 0.3125V
  
/* Status bit values */
#define AIN_SCAN_RUNNING   (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)

#define NCHAN_USB1608FS_PLUS     8  // max number of A/D channels in the device
#define NGAINS_USB1608FS_PLUS    8  // max number of gain levels
#define MAX_PACKET_SIZE         64  // max packet size for FS device

typedef struct t_calibrationTimeStamp {
  __u8 year;   // Calibration date year
  __u8 month;  // Calibration date month
  __u8 day;    // Calibration date day
  __u8 hour;   // Calibration date hour
  __u8 minute; // Calibration date minute
  __u8 second; // Calibration date second
} calibrationTimeStamp;

/* function prototypes for the USB-1608FS-Plus */
__u8 usbDTristateR_USB1608FS_Plus(usb_dev_handle *udev);
void usbDTristateW_USB1608FS_Plus(usb_dev_handle *udev, __u8 value);
__u8 usbDPort_USB1608FS_Plus(usb_dev_handle *udev);
__u8 usbDLatchR_USB1608FS_Plus(usb_dev_handle *udev);
 void usbDLatchW_USB1608FS_Plus(usb_dev_handle *udev, __u8 value);
__u16 usbAIn_USB1608FS_Plus(usb_dev_handle *udev, __u8 channel, __u8 range);
void usbAInScanStart_USB1608FS_Plus(usb_dev_handle *udev, __u32 count, double frequency, __u8 channels, __u8 options);
void usbAInScanConfig_USB1608FS_Plus(usb_dev_handle *udev, __u8 ranges[8]);
void usbAInScanConfigR_USB1608FS_Plus(usb_dev_handle *udev, __u8 *ranges);
int usbAInScanRead_USB1608FS_Plus(usb_dev_handle *udev, int nScan, int nChan, __u16 *data);
void usbAInScanStop_USB1608FS_Plus(usb_dev_handle *udev);
void usbAInScanClearFIFO_USB1608FS_Plus(usb_dev_handle *udev);
__u32 usbCounter_USB1608FS_Plus(usb_dev_handle *udev);
void usbCounterInit_USB1608FS_Plus(usb_dev_handle *udev);
void usbReadCalMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteCalMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbReadUserMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteUserMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbReadMBDMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteMBDMemory_USB1608FS_Plus(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbBlink_USB1608FS_Plus(usb_dev_handle *udev, __u8 count);
void usbReset_USB1608FS_Plus(usb_dev_handle *udev);
__u16 usbStatus_USB1608FS_Plus(usb_dev_handle *udev);
void usbGetSerialNumber_USB1608FS_Plus(usb_dev_handle *udev, char serial[9]);
void usbDFU_USB1608FS_Plus(usb_dev_handle *udev);
void usbMBDCommand_USB1608FS_Plus(usb_dev_handle *udev, __u8 str[]);
void usbMBDRaw_USB1608FS_Plus(usb_dev_handle *udev, __u8 cmd[], __u16 size);
void cleanup_USB1608FS_Plus(usb_dev_handle *udev);
void usbBuildGainTable_USB1608FS_Plus(usb_dev_handle *udev, float table[NGAINS_USB1608FS_PLUS][NCHAN_USB1608FS_PLUS][2]);
double volts_USB1608FS_Plus(usb_dev_handle *udev, __u16 value, __u8 range);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif //USB_1608FS_PLUS_H
