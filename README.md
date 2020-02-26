# Process to model the energy consumption of a fridge

# Strategy: 

Fridge's energy consumption is directly proportional to the time the compressor is on. Therefore, the aim is to model the length of the time the compressor is on.


Based on early impressions of the data (constrained by the time limit of 6 hours), it was decided to model the length of the time the compressor is on, based on 
1. the difference in temperatures inside the fridge when the door is opened and the temperature in the room (the latter was found to be the same as the user-defined setpoint, in most cases)
2. the length of time the door was open

This model would be coupled by another model which predicts the door open length at a given hour of the day.  

Once these model were developed, use the May 2019 hourly data of temperatures at LGA (https://www.ncdc.noaa.gov/cdo-web/datasets#NORMAL_HLY) in New York (it is assumed that the fridge is in New York, and that room temperature is equal to the outside temperature). 

The ipython notebooks in this repo show illustrate the the exploration and modelling, and the python scripts are used to automate the model building, and the actual prediction, for a given uuid and setpoint.

# Usage
 download the repo and run:
 
        python predict --uuid <uuid number> and --setpoint <setpoint>

# Assumptions/ Decisions:
1. Model can have 2 coupled parts: predicting compressor being on durations, and and predicting door open lengths
2. Each device is owned by different individual, who have their own model of keeping the door open, thereby, 


# Shortcomings/ Scope for improvement:
1. Fully automate the training pipeline - I was only able to automate the compressor-turned-on durations given door opening lengths and temperature differences, while I individually trained the probability durations for each uuid's door open length

2. Investigate LSTM for better time-series prediction 

3. Use facebook's prophet package for timeseries prediction
