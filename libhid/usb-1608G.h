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

#ifndef USB_1608G_H

#define USB_1608G_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB1608G_PID      (0x0110)
#define USB1608GX_PID     (0x0111)
#define USB1608GX_2AO_PID (0x0112)

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

/* Commands and HID Report ID for USB 1608G  */
/* Digital I/O Commands */
#define DTRISTATE     (0x00)   // Read/Write Tristate register
#define DPORT         (0x01)   // Read digital port pins
#define DLATCH        (0x02)   // Read/Write Digital port output latch register

/* Analog Input Commands */
#define AIN            (0x10)  // Read analog input channel
#define AIN_SCAN_START (0x12)  // Start input scan
#define AIN_SCAN_STOP  (0x13)  // Stop input scan
#define AIN_CONFIG     (0x14)  // Analog input channel configuration
#define AIN_CLR_FIFO   (0x15)  // Clear any remaining input in FIFO after scan.

/* Analog Output Commands  USB-1608GX-2A0 only*/
#define AOUT            (0x18)   // Read/Write analog output channel
#define AOUT_SCAN_START (0x1A)   // Start analog ouput scan
#define AOUT_SCAN_STOP  (0x1B)   // Stop analog output scan
#define AOUT_CLEAR_FIFO (0x1C)   // Clear data in analog output FIFO

/* Counter/Timer Commands */
#define COUNTER           (0x20)  // Read/reset event counter
#define TIMER_CONTROL     (0x28)  // Read/write timer control register
#define TIMER_PERIOD      (0x29)  // Read/write timer period register
#define TIMER_PULSE_WIDTH (0x2A)  // Read/write timer pulse width register
#define TIMER_COUNT       (0x2B)  // Read/write timer counter register
#define TIMER_START_DELAY (0x2C)  // Read/write timer start delay register
#define TIMER_PARAMETERS  (0x2D)  // Read/write timer parameters

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

/* Counter Timer */
#define COUNTER0         0x0     //  Counter 0
#define COUNTER1         0x1     //  Counter 1

/* Aanalog Input */
#define SINGLE_ENDED   0
#define DIFFERENTIAL   1
#define CALIBRATION    3
#define LAST_CHANNEL   (0x80)
#define PACKET_SIZE    512       // max bulk transfer size in bytes

/* Ananlog Output Scan Options */
#define AO_CHAN0       0x1   // Include Channel 0 in output scan
#define AO_CHAN1       0x2   // Include Channel 1 in output scan
#define AO_TRIG        0x10  // Use Trigger
#define AO_RETRIG_MODE 0x20  // Retrigger Mode
  
/* Ranges */
#define BP_10V 0x0      // +/- 10 V
#define BP_5V  0x1      // +/- 5V
#define BP_2V  0x2      // +/- 2V
#define BP_1V  0x3      // +/- 1V
  
/* Status bit values */
#define AIN_SCAN_RUNNING   (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)
#define AOUT_SCAN_RUNNING  (0x1 << 3)
#define AOUT_SCAN_UNDERRUN (0x1 << 4)
#define AIN_SCAN_DONE      (0x1 << 5)
#define AOUT_SCAN_DONE     (0x1 << 6)
#define FPGA_CONFIGURED    (0x1 << 8)
#define FPGA_CONFIG_MODE   (0x1 << 9)

#define NCHAN_1608G          16  // max number of A/D channels in the device
#define NGAINS_1608G          4  // max number of gain levels
#define NCHAN_AO_1608GX       2  // number of analog output channels
#define MAX_PACKET_SIZE_HS  512  // max packet size for HS device
#define MAX_PACKET_SIZE_FS   64  // max packet size for FS device

typedef struct t_timerParams {
  __u32 period;
  __u32 pulseWidth;
  __u32 count;
  __u32 delay;
} timerParams;

typedef struct t_ScanList {
  __u8 mode;
  __u8 range;
  __u8 channel;
} ScanList;

/* function prototypes for the USB-1608G */
void usbDTristateW_USB1608G(usb_dev_handle *udev, __u16 value);
__u16 usbDTristateR_USB1608G(usb_dev_handle *udev);
__u16 usbDPort_USB1608G(usb_dev_handle *udev);
void usbDLatchW_USB1608G(usb_dev_handle *udev, __u16 value);
__u16 usbDLatchR_USB1608G(usb_dev_handle *udev);
void usbBlink_USB1608G(usb_dev_handle *udev, __u8 count);
void cleanup_USB1608G( usb_dev_handle *udev);
void usbTemperature_USB1608G(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB1608G(usb_dev_handle *udev, char serial[9]);
void usbReset_USB1608G(usb_dev_handle *udev);
void usbFPGAConfig_USB1608G(usb_dev_handle *udev);
void usbFPGAData_USB1608G(usb_dev_handle *udev, __u8 *data, __u8 length);
void usbFPGAVersion_USB1608G(usb_dev_handle *udev, __u16 *version);
__u16 usbStatus_USB1608G(usb_dev_handle *udev);
void usbInit_1608G(usb_dev_handle *udev);
void usbCounterInit_USB1608G(usb_dev_handle *udev, __u8 counter);
__u32 usbCounter_USB1608G(usb_dev_handle *udev, __u8 counter);
void usbTimerControlR_USB1608G(usb_dev_handle *udev, __u8 *control);
void usbTimerControlW_USB1608G(usb_dev_handle *udev, __u8 control);
void usbTimerPeriodR_USB1608G(usb_dev_handle *udev, __u32 *period);
void usbTimerPeriodW_USB1608G(usb_dev_handle *udev, __u32 period);
void usbTimerPulseWidthR_USB1608G(usb_dev_handle *udev, __u32 *pulseWidth);
void usbTimerPulseWidthW_USB1608G(usb_dev_handle *udev, __u32 pulseWidth);
void usbTimerCountR_USB1608G(usb_dev_handle *udev, __u32 *count);
void usbTimerCountW_USB1608G(usb_dev_handle *udev, __u32 count);
void usbTimerDelayR_USB1608G(usb_dev_handle *udev, __u32 *delay);
void usbTimerDelayW_USB1608G(usb_dev_handle *udev, __u32 delay);
void usbTimerParamsR_USB1608G(usb_dev_handle *udev, timerParams *params);
void usbTimerParamsW_USB1608G(usb_dev_handle *udev, timerParams *params);
void usbMemoryR_USB1608G(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemoryW_USB1608G(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemAddressR_USB1608G(usb_dev_handle *udev, __u16 address);
void usbMemAddressW_USB1608G(usb_dev_handle *udev, __u16 address);
void usbMemWriteEnable_USB1608G(usb_dev_handle *udev);
void usbReset_USB1608G(usb_dev_handle *udev);
void usbTriggerConfig_USB1608G(usb_dev_handle *udev, __u8 options);
void usbTriggerConfigR_USB1608G(usb_dev_handle *udev, __u8 *options);
void usbTemperature_USB1608G(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB1608G(usb_dev_handle *udev, char serial[9]);
__u16 usbAIn_USB1608G(usb_dev_handle *udev, __u16 channel);
void usbAInScanStart_USB1608G(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 options);
void usbAInScanStop_USB1608G(usb_dev_handle *udev);
int usbAInScanRead_USB1608G(usb_dev_handle *udev, int nScan, int nChan, __u16 *data);
void usbAInConfig_USB1608G(usb_dev_handle *udev, ScanList scanList[NCHAN_1608G]);
int usbAInConfigR_USB1608G(usb_dev_handle *udev, __u8 *scanList);
void usbAInScanClearFIFO_USB1608G(usb_dev_handle *udev);
void usbBuildGainTable_USB1608G(usb_dev_handle *udev, float table[NGAINS_1608G][2]);
double volts_USB1608G(usb_dev_handle *udev, const __u8 gain, __u16 value);
void usbBuildGainTable_USB1608GX_2AO(usb_dev_handle *udev, float table_AO[NCHAN_AO_1608GX][2]);
__u16 voltsTou16_USB1608GX_AO(double volts, int channel, float table_AO[NCHAN_AO_1608GX][2]);
void usbAOut_USB1608GX_2AO(usb_dev_handle *udev, __u8 channel, double voltage, float table_AO[NCHAN_AO_1608GX][2]);
void usbAOutR_USB1608GX_2AO(usb_dev_handle *udev, __u8 channel, double *voltage, float table_AO[NCHAN_AO_1608GX][2]);
void usbAOutScanStop_USB1608GX_2AO(usb_dev_handle *udev);
void usbAOutScanClearFIFO_USB1608GX_2AO(usb_dev_handle *udev);
void usbAOutScanStart_USB1608GX_2AO(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 options);


#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif //USB_1608G_H
