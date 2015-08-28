/*
 *
 *  Copyright (c) 2014  Warren Jasper <wjasper@tx.ncsu.edu>
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

#ifndef USB_2020_H

#define USB_2020_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB2020_PID (0x011c)


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

/* Commands and HID Report ID for USB 2020  */
/* Digital I/O Commands */
#define DTRISTATE        (0x00)   // Read/Write Tristate register
#define DPORT            (0x01)   // Read digital port pins
#define DLATCH           (0x02)   // Read/Write Digital port output latch register

/* Analog Input Commands */
#define AIN              (0x10)  // Read analog input channel
#define AIN_SCAN_START   (0x12)  // Start input scan
#define AIN_SCAN_STOP    (0x13)  // Stop input scan
#define AIN_CONFIG       (0x14)  // Analog input channel configuration
#define AIN_CLR_FIFO     (0x15)  // Clear firmware analog input FIFO

/* Memory Commands */
#define MEMORY            (0x30)  // Read/Write EEPROM
#define MEM_ADDRESS       (0x31)  // EEPROM read/write address value
#define MEM_WRITE_ENABLE  (0x32)  // Enable writes to firmware area

/* Miscellaneous Commands */  
#define STATUS            (0x40)  // Read device status
#define BLINK_LED         (0x41)  // Causes LED to blink
#define RESET             (0x42)  // Reset device
#define TRIGGER_CONFIG    (0x43)  // External trigger configuration
#define CAL_CONFIG        (0x44)  // Calibration configuration
#define TEMPERATURE       (0x45)  // Read internal temperature
#define SERIAL            (0x48)  // Read/Write USB Serial Number

/* FPGA Configuration Commands */
#define FPGA_CONFIG       (0x50) // Start FPGA configuration
#define FPGA_DATA         (0x51) // Write FPGA configuration data
#define FPGA_VERSION      (0x52) // Read FPGA version

/* Aanalog Input */
#define SINGLE_ENDED   0
#define CALIBRATION    1
#define LAST_CHANNEL   (0x80)
#define PACKET_SIZE    512       // max bulk transfer size in bytes
  
/* Ranges */
#define BP_10V 0x0    // +/- 10 V
#define BP_5V  0x1    // +/- 5V
#define BP_2V  0x2    // +/- 2V
#define BP_1V  0x3    // +/- 1V

/* Options for AInScan*/
#define TRIGGER            (0x1 << 3) // 1 = use trigger or gate
#define PACER_OUT          (0x1 << 5) // 1 = External Pacer Output, 0 = External Pacer Input
#define RETRIGGER          (0x1 << 6) // 1 = retrigger mode, 0 = normal trigger
#define DDR_RAM            (0x1 << 7) // 1 = Use DDR RAM as storage, 0 = Stream via USB
  
/* Status bit values */
#define AIN_PACER_RUNNING  (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)
#define AIN_SCAN_DONE      (0x1 << 5)
#define FPGA_CONFIGURED    (0x1 << 8)
#define FPGA_CONFIG_MODE   (0x1 << 9)

#define NCHAN_2020            2 // max number of A/D channels in the device
#define NGAINS_2020           4 // max number of gain levels
#define MAX_PACKET_SIZE_HS  512 // max packet size for HS device
#define MAX_PACKET_SIZE_FS   64 // max packet size for FS device

typedef struct t_ScanList {
  __u8 channel;
  __u8 mode;
  __u8 range;
  __u8 last_channel;
} ScanList;

typedef struct t_TriggerConfig {
  __u8 options;
  __u8 triggerChannel;
  __u8 lowThreshold[2];
  __u8 highThreshold[2];
} TriggerConfig;

/* function prototypes for the USB-2020 */
void usbDTristateW_USB2020(usb_dev_handle *udev, __u16 value);
__u16 usbDTristateR_USB2020(usb_dev_handle *udev);
__u16 usbDPort_USB2020(usb_dev_handle *udev);
void usbDLatchW_USB2020(usb_dev_handle *udev, __u16 value);
__u16 usbDLatchR_USB2020(usb_dev_handle *udev);
void usbBlink_USB2020(usb_dev_handle *udev, __u8 count);
void cleanup_USB2020( usb_dev_handle *udev);
void usbTemperature_USB2020(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB2020(usb_dev_handle *udev, char serial[9]);
void usbReset_USB2020(usb_dev_handle *udev);
void usbCalConfig_USB2020(usb_dev_handle *udev, __u8 voltage);
void usbFPGAConfig_USB2020(usb_dev_handle *udev);
void usbFPGAData_USB2020(usb_dev_handle *udev, __u8 *data, __u8 length);
void usbFPGAVersion_USB2020(usb_dev_handle *udev, __u16 *version);
__u16 usbStatus_USB2020(usb_dev_handle *udev);
void usbInit_USB2020(usb_dev_handle *udev);
void usbMemoryR_USB2020(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemoryW_USB2020(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemAddressR_USB2020(usb_dev_handle *udev, __u16 address);
void usbMemAddressW_USB2020(usb_dev_handle *udev, __u16 address);
void usbMemWriteEnable_USB2020(usb_dev_handle *udev);
void usbTriggerConfig_USB2020(usb_dev_handle *udev, TriggerConfig *triggerConfig);
void usbTriggerConfigR_USB2020(usb_dev_handle *udev, TriggerConfig *triggerConfig);
void usbTemperature_USB2020(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB2020(usb_dev_handle *udev, char serial[9]);
__u16 usbAIn_USB2020(usb_dev_handle *udev, __u16 channel);
void usbAInScanStart_USB2020(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency,
			      __u32 packet_size, __u8 options);
void usbAInScanStop_USB2020(usb_dev_handle *udev);
int usbAInScanRead_USB2020(usb_dev_handle *udev, int nScan, int nChan, __u16 *data);
void usbAInConfig_USB2020(usb_dev_handle *udev, ScanList scanList[NCHAN_2020]);
void usbAInConfigR_USB2020(usb_dev_handle *udev, ScanList scanList[NCHAN_2020]);
void usbAInScanClearFIFO_USB2020(usb_dev_handle *udev);
void usbBuildGainTable_USB2020(usb_dev_handle *udev, float table[NGAINS_2020][2]);
double volts_USB2020(usb_dev_handle *udev, const __u8 gain, __u16 value);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif //USB_2020_H
