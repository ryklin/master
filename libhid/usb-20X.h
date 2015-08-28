/*
 *
 *  Copyright (c) 2009  Warren Jasper <wjasper@tx.ncsu.edu>
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

#ifndef USB_20X_H

#define USB_20X_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>
#define USB201_PID   (0x0113)
#define USB202_PID   (0x012b)
#define USB204_PID   (0x0114)
#define USB205_PID   (0x012c)

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

/* Commands and HID Report ID for USB 1208HS  */
/* Digital I/O Commands */
#define DTRISTATE            (0x00) // Read/write digital tristate register
#define DPORT                (0x01) // Read digital port pins
#define DLATCH               (0x02) // Read/write digital port output latch register

/* Analog Input Commands */
#define AIN                  (0x10) // Read analog input channel
#define AIN_SCAN_START       (0x11) // Start analog input scan
#define AIN_SCAN_STOP        (0x12) // Stop analog input scan
#define AIN_SCAN_CLR_FIFO    (0x15) // Clear the analog input scan FIFO
#define AIN_BULK_FLUSH       (0x16) // Flush the bulk endpoint with empty packets

/* Analog Output Commands (USB-202/205 only) */
#define AOUT                 (0x18) // Read/Write analog output channel

/* Counter Commands */
#define COUNTER              (0x20) // Read/reset event counter

/* Memory Commands */
#define CAL_MEMORY           (0x30) // Read/write calibration memory
#define USER_MEMORY          (0x31) // Read/write user memory
#define MBD_MEMORY           (0x32) // Read/write MBD memory
  
 /* Miscellaneous Commands */
#define BLINK_LED            (0x41) // Blink the LED
#define RESET                (0x42) // Reset the device
#define STATUS               (0x44) // Device Status
#define SERIAL               (0x48) // Read/write USB serial number
#define DFU                  (0x50) // Enter device firmware upgrade mode

/* MBD */
#define MBD_COMMAND          (0x80) // Text-based MBD command/response
#define MBD_RAW              (0x81) // Raw MBD response

/* Analog Input Scan Options */
#define IMMEDIATE_TRANSFER_MODE (0x1)
#define BLOCK_TRANSFER_MODE     (0x0)
#define STALL_ON_OVERRUN        (0x0)
#define INHIBIT_STALL           (0x1 << 7)
#define NO_TRIGGER              (0x0)
#define TRIGGER                 (0x1)
#define EDGE_RISING             (0x0)
#define EDGE_FALLING            (0x1)
#define LEVEL_HIGH              (0x2)
#define LEVEL_LOW               (0x3)
#define CHAN0                   (0x1)
#define CHAN1                   (0x2)
#define CHAN2                   (0x4)
#define CHAN3                   (0x8)
#define CHAN4                   (0x10)
#define CHAN5                   (0x20)
#define CHAN6                   (0x40)
#define CHAN7                   (0x80)
#define PACKET_SIZE             (64)
  
/* Status bit values */
#define AIN_SCAN_RUNNING   (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)

/* Calibration Coefficients */
#define NCHAN_USB20X 8

typedef struct t_calibrationTimeStamp {
  __u8 year;   // Calibration date year
  __u8 month;  // Calibration date month
  __u8 day;    // Calibration date day
  __u8 hour;   // Calibration date hour
  __u8 minute; // Calibration date minute
  __u8 second; // Calibration date second
} calibrationTimeStamp;

  
/* function prototypes for the USB-20X */
__u8 usbDTristateR_USB20X(usb_dev_handle *udev);
void usbDTristateW_USB20X(usb_dev_handle *udev, __u8 value);
__u8 usbDPort_USB20X(usb_dev_handle *udev);
__u8 usbDLatchR_USB20X(usb_dev_handle *udev);
void usbDLatchW_USB20X(usb_dev_handle *udev, __u8 value);

__u16 usbAIn_USB20X(usb_dev_handle *udev, __u8 channel);
void usbAOut_USB20X(usb_dev_handle *udev, __u8 channel, __u16 value);
void usbAInScanStart_USB20X(usb_dev_handle *udev, __u32 count, double frequency, __u8 channels, __u8 options, __u8 trigger_source, __u8 trigger_mode);
int usbAInScanRead_USB20X(usb_dev_handle *udev, int nScan, int nChan, __u16 *data);
void usbAInScanStop_USB20X(usb_dev_handle *udev);
void usbAInScanClearFIFO_USB20X(usb_dev_handle *udev);
void usbAInBulkFlush_USB20X(usb_dev_handle *udev, __u8 count);

__u32 usbCounter_USB20X(usb_dev_handle *udev);
void usbCounterInit_USB20X(usb_dev_handle *udev);
void usbReadCalMemory_USB20X(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteCalMemory_USB20X(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbReadUserMemory_USB20X(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteUserMemory_USB20X(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbReadMBDMemory_USB20X(usb_dev_handle *udev, __u16 address, __u16 count, __u8 memory[]);
void usbWriteMBDMemory_USB20X(usb_dev_handle *udev, __u16 address,  __u16 count, __u8 data[]);
void usbBlink_USB20X(usb_dev_handle *udev, __u8 count);
void usbReset_USB20X(usb_dev_handle *udev);
__u16 usbStatus_USB20X(usb_dev_handle *udev);
void usbGetSerialNumber_USB20X(usb_dev_handle *udev, char serial[9]);
void usbDFU_USB20X(usb_dev_handle *udev);
void usbMBDCommand_USB20X(usb_dev_handle *udev, __u8 str[]);
void usbMBDRaw_USB20X(usb_dev_handle *udev, __u8 cmd[]);
void cleanup_USB20X(usb_dev_handle *udev);
void usbBuildGainTable_USB20X(usb_dev_handle *udev, float table[NCHAN_USB20X][2]);
double volts_USB20X(usb_dev_handle *udev, __u16 value);
  

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif // USB_1208HS_H
