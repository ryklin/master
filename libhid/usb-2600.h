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

#ifndef USB_2600_H

#define USB_2600_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB2623_PID (0x0120)
#define USB2627_PID (0x0121)
#define USB2633_PID (0x0118)
#define USB2637_PID (0x0119)


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

/* Commands and HID Report ID for USB 2600  */
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

/* Analog Output Commands  USB-2600X-2A0 only*/
#define AOUT             (0x18)   // Read/Write analog output channel
#define AOUT_SCAN_START  (0x1A)   // Start analog ouput scan
#define AOUT_SCAN_STOP   (0x1B)   // Stop analog output scan
#define AOUT_CLEAR_FIFO  (0x1C)   // Clear data in analog output FIFO

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
#define COUNTER0         0x0     // Counter 0
#define COUNTER1         0x1     // Counter 1
#define COUNTER2         0x2     // Counter 2
#define COUNTER3         0x3     // Counter 3
#define TIMER0           0x0     // Timer 0 
#define TIMER1           0x1     // Timer 1 
#define TIMER2           0x2     // Timer 2 
#define TIMER3           0x3     // Timer 3 

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

/* Ananlog Output Scan Options */
#define AO_CHAN0       0x1   // Include Channel 0 in output scan
#define AO_CHAN1       0x2   // Include Channel 1 in output scan
#define AO_CHAN2       0x4   // Include Channel 2 in output scan
#define AO_CHAN3       0x8   // Include Channel 3 in output scan
#define AO_TRIG        0x10  // Use Trigger
#define AO_RETRIG_MODE 0x20  // Retrigger Mode
  
/* Status bit values */
#define AIN_SCAN_RUNNING   (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)
#define AOUT_SCAN_RUNNING  (0x1 << 3)
#define AOUT_SCAN_UNDERRUN (0x1 << 4)
#define AIN_SCAN_DONE      (0x1 << 5)
#define AOUT_SCAN_DONE     (0x1 << 6)
#define FPGA_CONFIGURED    (0x1 << 8)
#define FPGA_CONFIG_MODE   (0x1 << 9)

#define NCHAN_2600           64 // max number of A/D channels in the device
#define NGAINS_2600           4 // max number of gain levels
#define NCHAN_AO_26X7         4 // number of analog output channels 
#define MAX_PACKET_SIZE_HS  512 // max packet size for HS device
#define MAX_PACKET_SIZE_FS   64 // max packet size for FS device

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

/* function prototypes for the USB-2600 */
void usbDTristateW_USB2600(usb_dev_handle *udev, __u8 port, __u16 value);
__u16 usbDTristateR_USB2600(usb_dev_handle *udev, __u8 port);
__u16 usbDPort_USB2600(usb_dev_handle *udev, __u8 port);
void usbDLatchW_USB2600(usb_dev_handle *udev,__u8 port,  __u16 value);
__u16 usbDLatchR_USB2600(usb_dev_handle *udev, __u8 port);
void usbBlink_USB2600(usb_dev_handle *udev, __u8 count);
void cleanup_USB2600( usb_dev_handle *udev);
void usbTemperature_USB2600(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB2600(usb_dev_handle *udev, char serial[9]);
void usbReset_USB2600(usb_dev_handle *udev);
void usbFPGAConfig_USB2600(usb_dev_handle *udev);
void usbFPGAData_USB2600(usb_dev_handle *udev, __u8 *data, __u8 length);
void usbFPGAVersion_USB2600(usb_dev_handle *udev, __u16 *version);
__u16 usbStatus_USB2600(usb_dev_handle *udev);
void usbInit_2600(usb_dev_handle *udev);
void usbCounterInit_USB2600(usb_dev_handle *udev, __u8 counter);
__u32 usbCounter_USB2600(usb_dev_handle *udev, __u8 counter);
void usbTimerControlR_USB2600(usb_dev_handle *udev, __u8 timer, __u8 *control);
void usbTimerControlW_USB2600(usb_dev_handle *udev, __u8 timer, __u8 control);
void usbTimerPeriodR_USB2600(usb_dev_handle *udev, __u8 timer, __u32 *period);
void usbTimerPeriodW_USB2600(usb_dev_handle *udev, __u8 timer, __u32 period);
void usbTimerPulseWidthR_USB2600(usb_dev_handle *udev, __u8 timer, __u32 *pulseWidth);
void usbTimerPulseWidthW_USB2600(usb_dev_handle *udev, __u8 timer, __u32 pulseWidth);
void usbTimerCountR_USB2600(usb_dev_handle *udev, __u8 timer, __u32 *count);
void usbTimerCountW_USB2600(usb_dev_handle *udev, __u8 timer, __u32 count);
void usbTimerDelayR_USB2600(usb_dev_handle *udev, __u8 timer, __u32 *delay);
void usbTimerDelayW_USB2600(usb_dev_handle *udev, __u8 timer, __u32 delay);
void usbTimerParamsR_USB2600(usb_dev_handle *udev, __u8 timer, timerParams *params);
void usbTimerParamsW_USB2600(usb_dev_handle *udev, __u8 timer, timerParams *params);
void usbMemoryR_USB2600(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemoryW_USB2600(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemAddressR_USB2600(usb_dev_handle *udev, __u16 address);
void usbMemAddressW_USB2600(usb_dev_handle *udev, __u16 address);
void usbMemWriteEnable_USB2600(usb_dev_handle *udev);
void usbTriggerConfig_USB2600(usb_dev_handle *udev, __u8 options);
void usbTriggerConfigR_USB2600(usb_dev_handle *udev, __u8 *options);
void usbTemperature_USB2600(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB2600(usb_dev_handle *udev, char serial[9]);
__u16 usbAIn_USB2600(usb_dev_handle *udev, __u16 channel);
void usbAInScanStart_USB2600(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency,
			      __u8 packet_size, __u8 options);
void usbAInScanStop_USB2600(usb_dev_handle *udev);
int usbAInScanRead_USB2600(usb_dev_handle *udev, int nScan, int nChan, __u16 *data);
void usbAInConfig_USB2600(usb_dev_handle *udev, ScanList scanList[NCHAN_2600]);
void usbAInConfigR_USB2600(usb_dev_handle *udev, ScanList scanList[NCHAN_2600]);
void usbAInScanClearFIFO_USB2600(usb_dev_handle *udev);
void usbAOut_USB26X7(usb_dev_handle *udev, __u8 channel, double voltage, float table_AO[NCHAN_AO_26X7][2]);
void usbAOutR_USB26X7(usb_dev_handle *udev, __u8 channel, double *voltage, float table_AO[NCHAN_AO_26X7][2]);
void usbAOutScanStop_USB26X7(usb_dev_handle *udev);
void usbAOutScanClearFIFO_USB26X7(usb_dev_handle *udev);
void usbAOutScanStart_USB2600(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 options);
void usbBuildGainTable_USB2600(usb_dev_handle *udev, float table[NGAINS_2600][2]);
void usbBuildGainTable_USB26X7(usb_dev_handle *udev, float table_AO[NCHAN_AO_26X7][2]);
void usbAOut_USB26X7(usb_dev_handle *udev, __u8 channel, double voltage, float table_AO[NCHAN_AO_26X7][2]);
void usbAOutScanStop_USB26X7(usb_dev_handle *udev);
void usbAOutScanClearFIFO_USB26X7(usb_dev_handle *udev);
void usbAOutScanStart_USB26X7(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 options);
double volts_USB2600(usb_dev_handle *udev, const __u8 gain, __u16 value);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif //USB_2600_H
