# ASEN 6519: Project 3
Spring 2022
## Background
In this assignment, we will explore open-loop GPS signal tracking using the same mountaintop dataset from projects 1 and 2. In the previous project, you ran a closed tracking loop to estimate signal code and carrier phase parameters over the duration of the file. In this project, we will focus on obtaining estimates of the signal parameters, i.e., amplitude, code phase, and carrier phase, using open-loop tracking.

## Objectives
In this assignment you will:
- Explore the difference between open and closed loop tracking.
- Implement open-loop tracking for direct line-of-sight and coherent ocean reflection signals.
- Generate signal models for open loop tracking.

## Instructions
1. Raw IF data file: haleakala_20210611_160000_RX7.dat
2. Signal Model: haleakala_20210611_160000_RX7_signal_model.mat

| **Variables**            | **Units**       | **Descriptions**                                                                                                  |
|--------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------|
| _gpsweek_                | --              | GPS Week                                                                                                          |
| _timeVec_                | sec             | GPS second of signal receiving time                                                                               |
| _tau_D, tau_R_           | Chip            | Code phase models of direct and reflected signals, assuming the specular point (SP) being on the mean sea surface |
| _doppler_D, doppler_R_   | Hz              | Doppler models of direct and reflected signals                                                                    |
| _el_dlos, az_dlos_       | deg             | GPS satellite azimuth and elevation angles at receiver                                                            |
| _l_sp, az_sp_            | deg             | GPS satellite azimuth and elevation angles at SP                                                                  |
| _sp_lat, sp_lat, sp_mss_ | deg, deg, meter | SP position in WGS84 reference frame                                                                              |
3. Navigation Solution : haleakala_20210611_160000_RX7_nav.mat

| **Variables**       | **Units**       | **Descriptions**                      |
|---------------------|-----------------|---------------------------------------|
| Rx_TimeStamp        | sec             | Receiver timestamps with clock bias   |
| Rx_X, Rx_Y, Rx_Z    | meter           | Receiver position in ECEF coordinate  |
| Rx_Vx, Rx_Vy, Rx_Vz | m/s             | Receiver velocity in ECEF coordinate  |
| Rx_Clk_Bias         | sec             | Receiver clock bias                   |
| Rx_Clk_Drift        | s/s             | Receiver clock drift                  |
| Rx_lat, Rx_lon,     | deg, deg, meter | Receiver position in WGS84 coordinate |

4. DTU18 Mean Sea Surface (MSS) model: [dtu18.mat](https://www.space.dtu.dk/english/Research/Scientific_data_and_models/Global_Mean_sea_surface)
5. Some utility scripts, e.g., specular point estimation. To be updated.
## Tasks
1. Modify the closed-loop tracking program that has been used for Project 2 to conduct open loop tracking of PRN 5 direct line-of-sight signal. The modification is mainly to generate the reference signal using the provided signal models in ‘haleakala_20210611_160000_RX7_signal_model.mat’, instead of using the signal parameter estimates from the closed-loop feedback. Also use the signal models to detrend code phase and carrier phase measurements from both closed and open loop tracking and compare the detrended results in two subplots. (40 Points).
2. Similar to task 1, track the PRN 5 reflection signal.
The signal navigation data bits can be obtained from closed-loop tracking of the PRN 5 direct signal. Remove the data bits from the open loop tracking correlation outputs of PRN 5 reflected signal. After removing the data bits, a four-quadrant carrier phase discriminator can be applied. Plot the two-quadrant and four-quadrant (with data bits removed) carrier phase discriminator outputs in two subplots. (30 Points) 
3. Generate the code phase and Doppler models using the provided receiver navigation solution ‘haleakala_20210611_160000_RX7_nav.mat’, and compare your model with the provided model for PRN 5 direct line-of-sight signal. Show the differences for code phase and Doppler frequency (make sure to clearly label the axes with units). The provided receiver navigation solution is obtained from the raw data using closed-loop tracking and least square estimation, so it is noisy. You may consider applying any filtering, smoothing, or fitting methods to any of its variables, including the estimated receiver timestamps. (20 Points)
4. Challenging problem: Similar to task 3, generate the signal code phase and Doppler frequency model for the PRN 5 reflected signal, compare with the provided model, and plot their differences. Here you are also required to calculate the reflection specular point based on the provided receiver navigation solution ‘haleakala_20210611_160000_RX7_nav.mat’ and the DTU18 mean sea surface model. (10 Points)
