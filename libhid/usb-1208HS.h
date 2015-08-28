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

#ifndef USB_1208HS_H

#define USB_1208HS_H
#ifdef __cplusplus
extern "C" { 
#endif

#include <usb.h>

#define USB1208HS_PID     (0x00c4)
#define USB1208HS_2AO_PID (0x00c5)
#define USB1208HS_4AO_PID (0x00c6)

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
#define AIN_SCAN_START       (0x12) // Start analog input scan
#define AIN_SCAN_STOP        (0x13) // Stop analog input scan
#define AIN_CONFIG           (0x14) // Analog input configuration

/* Analog Output Commands */
#define AOUT                 (0x18) // Read/write analog output channel
#define AOUT_SCAN_START      (0x1a) // Start analog output scan
#define AOUT_SCAN_STOP       (0x1b) // Stop analog output scan
#define AOUT_CLEAR_FIFO      (0x1c) // Clear data in analog ouptut FIFO

/* Counter/Timer Commands */
#define COUNTER              (0x20) // Read/reset event counter
#define TIMER_CONTROL        (0x28) // Read/write timer control register
#define TIMER_PERIOD         (0x29) // Read/write timer period register
#define TIMER_PULSE_WIDTH    (0x2a) // Read/write timer pulse width register
#define TIMER_COUNT          (0x2b) // Read/write timer count register
#define TIMER_START_DELAY    (0x2c) // Read/write timer start delay register
#define TIMER_PARAMETERS     (0x2d) // Read/write timer parameters

/* Memory Commands */
#define MEMORY               (0x30) // Read/write EEPROM
#define MEM_ADDRESS          (0x31) // Read/write EEPROM address value
#define MEM_WRITE_ENABLE     (0x32) // Enable writes to firmware area

/* Miscellaneous Commands */
#define STATUS               (0x40) // Device Status
#define BLINK_LED            (0x41) // Blink the LED
#define RESET                (0x42) // Reset the device
#define TRIGGER_CONFIG       (0x43) // External trigger configuration
#define TEMPERATURE          (0x45) // Read internal temperature
#define SERIAL               (0x48) // Read/write USB serial number
#define FPGA_CONFIG          (0x50) // Start FPGA configuration
#define FPGA_DATA            (0x51) // Write FPGA configuration data
#define FPGA_VERSION         (0x52) // Read FPGA version

/* Status bit values */
#define AIN_SCAN_RUNNING   (0x1 << 1)
#define AIN_SCAN_OVERRUN   (0x1 << 2)
#define AOUT_SCAN_RUNNING  (0x1 << 3)
#define AOUT_SCAN_UNDERRUN (0x1 << 4)
#define AIN_SCAN_DONE      (0x1 << 5)
#define AOUT_SCAN_DONE     (0x1 << 6)
#define FPGA_CONFIGURED    (0x1 << 8)
#define FPGA_CONFIG_MODE   (0x1 << 9)

/* Counter Timer */
#define COUNTER0         0x0      //  Counter 0
#define COUNTER1         0x1      //  Counter 1

#define NCHAN_1208HS          8   // max number of A/D channels in the device
#define NGAINS_1208HS         4   // max number of gain levels (analog input)
#define NMODE                 4   // max number of configuration modes
#define NCHAN_AO_1208HS       4   // number of analog output channels
#define MAX_PACKET_SIZE_HS  512   // max packet size for HS device
#define MAX_PACKET_SIZE_FS   64   // max packet size for FS device


/* Analog Input Scan and Modes */
#define SINGLE_ENDED        0     // 8 single-ended inputs
#define PSEUDO_DIFFERENTIAL 1     // 4 pseudo differential inputs
#define DIFFERENTIAL        2     // 4 true differential inputs
#define PSEUDO_DIFFERENTIAL_UP    // 7 pseudo differential inputs
#define PACKET_SIZE         512   // max bulk transfer size in bytes

/* Analog Input Scan Options */
#define CHAN0  (0x1 << 0)
#define CHAN1  (0x1 << 1)
#define CHAN2  (0x1 << 2)
#define CHAN3  (0x1 << 3)
#define CHAN4  (0x1 << 4)
#define CHAN5  (0x1 << 5)
#define CHAN6  (0x1 << 6) 
#define CHAN7  (0x1 << 7)

#define BURST_MODE   0x1
#define TRIGGER_MODE 0x8
#define DEBUG_MODE   0x20
#define RETRIG_MODE  0x40

#define BP_10V   0
#define BP_5V    1
#define BP_2_5V  2
#define UP_10V   3

#define BP_20V_DE 0
#define BP_10V_DE 1
#define BP_5V_DE  2

/* Ananlog Output Scan Options */
#define AO_CHAN0       0x1   // Include Channel 0 in output scan
#define AO_CHAN1       0x2   // Include Channel 1 in output scan
#define AO_CHAN2       0x4   // Include Channel 2 in output scan
#define AO_CHAN3       0x8   // Include Channel 3 in output scan
#define AO_TRIG        0x10  // Use Trigger
#define AO_RETRIG_MODE 0x20  // Retrigger Mode

typedef struct t_timerParams {
  __u32 period;
  __u32 pulseWidth;
  __u32 count;
  __u32 delay;
} timerParams;

/* function prototypes for the USB-1208HS */
__u16  usbDTristateR_USB1208HS(usb_dev_handle *udev);
void usbDTristateW_USB1208HS(usb_dev_handle *udev, __u16 value);
__u16 usbDPort_USB1208HS(usb_dev_handle *udev);
__u16 usbDLatchR_USB1208HS(usb_dev_handle *udev);
void usbDLatchW_USB1208HS(usb_dev_handle *udev, __u16 data);
void cleanup_USB1208HS(usb_dev_handle *udev);
void usbBlink_USB1208HS(usb_dev_handle *udev, __u8 count);
void usbTemperature_USB1208HS(usb_dev_handle *udev, float *temperature);
void usbGetSerialNumber_USB1208HS(usb_dev_handle *udev, char serial[9]);
void usbReset_USB1208HS(usb_dev_handle *udev);
void usbFPGAConfig_USB1208HS(usb_dev_handle *udev);
void usbFPGAData_USB1208HS(usb_dev_handle *udev, __u8 *data, __u8 length);
void usbFPGAVersion_USB1208HS(usb_dev_handle *udev, __u16 *version);
__u16 usbStatus_USB1208HS(usb_dev_handle *udev);
void usbMemoryR_USB1208HS(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemoryW_USB1208HS(usb_dev_handle *udev, __u8 *data, __u16 length);
void usbMemAddressR_USB1208HS(usb_dev_handle *udev, __u16 address);
void usbMemAddressW_USB1208HS(usb_dev_handle *udev, __u16 address);
void usbMemWriteEnable_USB1208HS(usb_dev_handle *udev);
void usbTriggerConfig_USB1208HS(usb_dev_handle *udev, __u8 options);
void usbTriggerConfigR_USB1208HS(usb_dev_handle *udev, __u8 *options);
void usbInit_1208HS(usb_dev_handle *udev);
void usbCounterInit_USB1208HS(usb_dev_handle *udev, __u8 counter);
__u32 usbCounter_USB1208HS(usb_dev_handle *udev, __u8 counter);
void usbTimerControlR_USB1208HS(usb_dev_handle *udev, __u8 *control);
void usbTimerControlW_USB1208HS(usb_dev_handle *udev, __u8 control);
void usbTimerPeriodR_USB1208HS(usb_dev_handle *udev, __u32 *period);
void usbTimerPeriodW_USB1208HS(usb_dev_handle *udev, __u32 period);
void usbTimerPulseWidthR_USB1208HS(usb_dev_handle *udev, __u32 *pulseWidth);
void usbTimerPulseWidthW_USB1208HS(usb_dev_handle *udev, __u32 pulseWidth);
void usbTimerCountR_USB1208HS(usb_dev_handle *udev, __u32 *count);
void usbTimerCountW_USB1208HS(usb_dev_handle *udev, __u32 count);
void usbTimerDelayR_USB1208HS(usb_dev_handle *udev, __u32 *delay);
void usbTimerDelayW_USB1208HS(usb_dev_handle *udev, __u32 delay);
void usbTimerParamsR_USB1208HS(usb_dev_handle *udev, timerParams *params);
void usbTimerParamsW_USB1208HS(usb_dev_handle *udev, timerParams *params);
__u16 usbAIn_USB1208HS(usb_dev_handle *udev, __u8 channel);
void usbAInConfig_USB1208HS(usb_dev_handle *udev, __u8 mode, __u8 range[NCHAN_1208HS]);
void usbAInConfigR_USB1208HS(usb_dev_handle *udev, __u8 *mode, __u8 range[NCHAN_1208HS]);
void usbAInScanStop_USB1208HS(usb_dev_handle *udev);
void usbAInScanStart_USB1208HS(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 channels, __u8 packet_size, __u8 options);
int usbAInScanRead_USB1208HS(usb_dev_handle *udev, int nScan, int nChan,  __u16 *data);
void usbAOut_USB1208HS(usb_dev_handle *udev, __u8 channel, double voltage, float table_AO[NCHAN_AO_1208HS][2]);
void usbAOutR_USB1208HS(usb_dev_handle *udev, __u8 channel, double *voltage, float table_AO[NCHAN_AO_1208HS][2]);
void usbAOutScanStop_USB1208HS(usb_dev_handle *udev);
void usbAOutScanClearFIFO_USB1208HS(usb_dev_handle *udev);
void usbAOutScanStart_USB1208HS(usb_dev_handle *udev, __u32 count, __u32 retrig_count, double frequency, __u8 options);
void usbBuildGainTable_USB1208HS(usb_dev_handle *udev, float table[NMODE][NGAINS_1208HS][2]);
void usbBuildGainTable_USB1208HS_4AO(usb_dev_handle *udev, float table_AO[NCHAN_AO_1208HS][2]);
__u16 voltsTou12_USB1208HS_AO(double volts, int channel, float table_AO[NCHAN_AO_1208HS][2]);
double volts_USB1208HS(usb_dev_handle *udev, const __u8 mode, const __u8 gain, __u16 value);  

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif
#endif // USB_1208HS_H
