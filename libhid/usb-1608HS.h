/*
 *
 *  Copyright (c) 2008  Warren Jasper <wjasper@tx.ncsu.edu>
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

#ifndef USB_1608HS_H

#define USB_1608HS_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB1608HS_PID (0x00bd)
#define USB1608HS_2AO_PID (0x0099)

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

/* Commands and HID Report ID for USB 1608HS  */
/* Digital I/O Commands */
#define DIN           (0x00)     // Read digital port
#define DIN_BIT       (0x01)     // Read digital port bit
#define DOUT          (0x08)     // Write digital port
#define DOUT_BIT      (0x09)     // Write digital port bit

/* Analog Input Commands */
#define AIN           (0x10)     // Read analog input channel
#define AIN_SCAN_CFG  (0x11)     // Analog input scan configuration
#define AIN_START     (0x12)     // Start input scan
#define AIN_STOP      (0x13)     // Stop input scan
#define AIN_CONFIG    (0x14)     // Analog input channel configuration

/* Analog Output Commands */
#define AOUT          (0x18)     // Write analog output channel
#define AOUT_SCAN_CFG (0x19)     // Analog output scan configuration
#define AOUT_START    (0x1A)     // Start analog ouput scan
#define AOUT_STOP     (0x1B)     // Stop analog output scan
#define AOUT_CONFIG   (0x1D)     // Analog output channel configuration

/* Miscellaneous Commands */
#define COUNTER     (0x20)     // Counter Value
#define MEMORY      (0x30)     // Read/Write EEPROM
#define MEM_ADDRESS (0x31)     // EEPROM read/write address value
#define STATUS      (0x40)     // Read device status
#define BLINK_LED   (0x41)     // Causes LED to blink
#define RESET       (0x42)     // Reset device
#define TRIGGER_CFG (0x43)     // External trigger configuration
#define CAL_CONFIG  (0x44)     // Calibration configuration
#define TEMPERATURE (0x45)     // Read internal temperature
#define SERIAL      (0x48)     // Read/Write USB Serial Number

/* Code Update Commands */
#define UPDATE_MODE      (0x50) // Put device into update mode
#define UPDATE_ADDRESS   (0x51) // Update address
#define UPDATE_DATA      (0x52) // Update data
#define UPDATE_VERSION   (0x53) // Update code version
#define UPDATE_CHECKSUM  (0x54) // Read/reset code update checksum

#define EXT_TRIG_FAILING_EDGE 0;
#define EXT_TRIG_RAISING_EDGE 1;

#define DIFFERENTIAL (0x0)
#define SINGLE_ENDED (0x1)
#define CALIBRATION  (0x2)
#define GROUND       (0x3 << 2)

// Gain Ranges
#define SE_10_00V  (0x0 | (SINGLE_ENDED << 2))    // Single Ended 0 - 10.0 V
#define SE_5_00V   (0x1 | (SINGLE_ENDED << 2))    // Single Ended 0 - 5.00 V
#define SE_2_00V   (0x2 | (SINGLE_ENDED << 2))    // Single Ended 0 - 2.00 V
#define SE_1_00V   (0x3 | (SINGLE_ENDED << 2))    // Single Ended 0 - 1.00 V
#define DE_10_00V  (0x0)                          // Differential +/- 10.0 V
#define DE_5_00V   (0x1)                          // Differential +/- 5.00 V
#define DE_2_00V   (0x2)                          // Differential +/- 2.00 V
#define DE_1_00V   (0x3)                          // Differential +/- 1.00 V

// Option values for AInScan
#define AIN_SINGLE_MODE    0x1  // 1 = single execution, 0 = continuous execution
#define AIN_BURST_MODE     0x2  // 1 = burst I/O mode (single execution only),   0 = normal I/O mode
#define AIN_TRANSFER_MODE  0x4  // 1 = Immediate Transfer mode (N/A in burst mode)  0 = block transfer mode
#define AIN_TRIGGER        0x8  // 1 = Use External Trigger
#define AIN_EXTERN_SYNC    0x10 // 1 = Use External Sync
#define AIN_DEBUG_MODE     0x20 // 1 = debug mode 0 = normal data
#define AIN_RETRIGGER_MODE 0x40 // 1 = retrigger mode, 0 = normal trigger
#define AIN_DIN_DATA       0x80 // 1 = include DIn data with each scan

// Device Status
#define DEVICE_TRIGGERED     (0x1)
#define INPUT_SCAN_RUNNING   (0x2)
#define INPUT_SCAN_OVERRUN   (0x4)
#define OUTPUT_SCAN_RUNNING  (0x8)
#define OUTPUT_SCAN_UNDERRUN (0x10)
#define INPUT_SCAN_COMPLETE  (0x20)
#define OUTPUT_SCAN_COMPLETE (0x40)

#define NCHAN_1608HS          8  // max number of A/D channels in the device
#define NGAINS_1608HS         4  // max number of gain levels
#define MAX_PACKET_SIZE_HS  512  // max packet size for HS device
#define MAX_PACKET_SIZE_FS   64  // max packet size for FS device

/* function prototypes for the USB-1608HS */
__u8 usbDIn_USB1608HS(usb_dev_handle *udev);
__u8 usbDInBit_USB1608HS(usb_dev_handle *udev, __u8 bit_num);
void usbDOut_USB1608HS(usb_dev_handle *udev, __u8 value);
void usbDOutBit_USB1608HS(usb_dev_handle *udev, __u8 bit_num, __u8 value);
void usbBlink_USB1608HS(usb_dev_handle *udev, __u8 count);
void usbAInConfig_USB1608HS(usb_dev_handle *udev,  __u8 config_array[8]);
void usbAInConfigRead_USB1608HS(usb_dev_handle *udev,  __u8 config_array[8]);
void usbAInScanStart_USB1608HS(usb_dev_handle *udev);
void usbAInScanStop_USB1608HS(usb_dev_handle *udev);
int usbAInScan_USB1608HS(usb_dev_handle *udev, __u16 *data);
int usbAIn_USB1608HS(usb_dev_handle *udev, __u16 *data);
void usbAInScanConfig_USB1608HS(usb_dev_handle *udev, __u8 lowChan, __u8 numChan, int count, float freq, __u8 options);
__u8 usbStatus_USB1608HS(usb_dev_handle *udev);
void usbReset_USB1608HS(usb_dev_handle *udev);
void usbTrigger_USB1608HS(usb_dev_handle *udev, __u16 data, __u8 options);
float usbTemperature_(usb_dev_handle *udev);
void usbGetSerialNumber_USB1608HS(usb_dev_handle *udev, char serial[9]);
void usbSetSerialNumber_USB1608HS(usb_dev_handle *udev, char serial[9]);
void usbReadMemory_USB1608HS(usb_dev_handle *udev, __u16 length, __u8 *data);
void usbWriteMemory_USB1608HS(usb_dev_handle *udev, __u16 length, __u8 *data);
void usbGetMemAddress_USB1608HS(usb_dev_handle *udev, __u16 *address);
void usbSetMemAddress_USB1608HS(usb_dev_handle *udev, __u16 address);
void usbAOut_USB1608HS(usb_dev_handle *udev, __u16 data[2]);
void usbAOutRead_USB1608HS(usb_dev_handle *udev, __u16 *data);
void usbAOutScanStart_USB1608HS(usb_dev_handle *udev);
void usbAOutScanStop_USB1608HS(usb_dev_handle *udev);
void usbAOutConfig_USB1608HS(usb_dev_handle *udev,  __u8 config);
void usbAOutConfigRead_USB1608HS(usb_dev_handle *udev, __u8 *config);
void usbCounterInit_USB1608HS(usb_dev_handle *udev);
void usbAOutScanConfig_USB1608HS(usb_dev_handle *udev, __u32 nscans, float frequency, __u8 options);
int usbAOutScan_USB1608HS(usb_dev_handle *udev, __u16 *data, __u32 ndata);
__u32 usbCounter_USB1608HS(usb_dev_handle *udev);
void usbUpdateMode_USB1608HS(usb_dev_handle *udev, __u8 device);
void usbUpdateAddress_USB1608HS(usb_dev_handle *udev, __u32 address);
void usbUpdateData_USB1608HS(usb_dev_handle *udev, __u16 length, __u8 *data);
__u16 usbUpdateVersion_USB1608HS(usb_dev_handle *udev);
__u16 usbUpdateChecksum_USB1608HS(usb_dev_handle *udev);
void usbUpdateChecksumReset_USB1608HS(usb_dev_handle *udev);
void usbBuildGainTable_USB1608HS(usb_dev_handle *udev, float *table[NCHAN_1608HS][NGAINS_1608HS][2]);
float volts_USB1608HS(usb_dev_handle *udev, const int channel, const int gain, const __u16 value);
void cleanup_USB1608HS(usb_dev_handle *udev);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif //USB_1608HS_H
