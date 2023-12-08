import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import csv

### HEADING ###
# AA228
# Final project
# Author: Matthew Tan
# Description: This function generates the data for Q learning according to the format (s,a,r,s').
# It sets an L1 norm based reward shaping gradient towards the specificed destination point.
# It adds and subtracts reward based on the airports and bad weather along the way
# It removes data from the resulting data table if the action is infeasible (such that the agent would have never seen it)
###############


### HELPER FUNCTIONS ####
def flat_index(length, width, x, y):
    return x + length*y + 1

def inject_reward(x,y,new_reward):
    if x > 0:
        (state,action,reward,next_state) = world[0,x-1,y]
        world[0,x-1,y] = (state,action,new_reward,next_state)

    if x < width-1:
        (state,action,reward,next_state) = world[2,x+1,y]
        world[2,x+1,y] = (state,action,new_reward,next_state)

    if y > 0:
        (state,action,reward,next_state) = world[1,x,y-1]
        world[1,x,y-1] = (state,action,new_reward,next_state)

    if y < length-1:
        (state,action,reward,next_state) = world[3,x,y+1]
        world[3,x,y+1] = (state,action,new_reward,next_state)

def add_reward(x,y,new_reward):
    if x > 0:
        (state,action,reward,next_state) = world[0,x-1,y]
        world[0,x-1,y] = (state,action,reward+new_reward,next_state)

    if x < width-1:
        (state,action,reward,next_state) = world[2,x+1,y]
        world[2,x+1,y] = (state,action,reward+new_reward,next_state)

    if y > 0:
        (state,action,reward,next_state) = world[1,x,y-1]
        world[1,x,y-1] = (state,action,reward+new_reward,next_state)

    if y < length-1:
        (state,action,reward,next_state) = world[3,x,y+1]
        world[3,x,y+1] = (state,action,reward+new_reward,next_state)
    

##########################################################
#
#
### CODE START ###

#state action reward sprime
state_type = np.dtype([('state', 'i4'), ('action', 'i4'), ('reward', 'i4'), ('s_prime', 'i4')])
width = 10
length = 10
impossible_reward = -999
destination = (length-1,width-1)
#destination = (0,2)
destination_reward = 100
airport_reward = 40
weather_reward = -70
# world is mapped (action, width coord, length coord)
world = np.empty((4,width,length), state_type)


### WORLD SPACE ###
# Top left is 0,0
####################

### ACTION SPACE ###
#    3
# 2  x  0
#    1
####################

# Generate basic action table
for (x,col) in enumerate(world[0,:]):
    for (y,state) in enumerate(col):
        if x < width-1:
            world[0,x,y] = (flat_index(length, width, x,y), 1, 0, flat_index(length, width, x+1, y))
        else:
            world[0,x,y] = (flat_index(length, width, x,y), 1, impossible_reward, flat_index(length, width, x, y))
            
        if y < length-1:
            world[1,x,y] = (flat_index(length, width, x,y), 2, 0, flat_index(length, width, x, y+1))
        else:
            world[1,x,y] = (flat_index(length, width, x,y), 2, impossible_reward, flat_index(length, width, x, y))
            
        if x > 0:
            world[2,x,y] = (flat_index(length, width, x,y), 3, 0, flat_index(length, width, x-1, y))
        else:
            world[2,x,y] = (flat_index(length, width, x,y), 3, impossible_reward, flat_index(length, width, x, y))
            
        if y > 0:
            world[3,x,y] = (flat_index(length, width, x,y), 4, 0, flat_index(length, width, x, y-1))
        else:
            world[3,x,y] = (flat_index(length, width, x,y), 4, impossible_reward, flat_index(length, width, x, y))


# Find max distance
distances = []
for x in range(width):
    for y in range(length):
        distance = np.abs(x-destination[0]) + np.abs(y-destination[1])
        distances.append(distance)
max_distance = max(distances)

# Create L1 norm gradient
for x in range(width):
    for y in range(length):
        for a in range(4):
            distance = np.abs(x-destination[0]) + np.abs(y-destination[1])
            inject_reward(x,y, destination_reward * (max_distance-distance)/max_distance)

# Add and subtract rewards based on weather
add_reward(destination[0],destination[1],airport_reward)
#add_reward(0,0, weather_reward)

add_reward(4,0,weather_reward)
add_reward(4,0,weather_reward)
add_reward(4,1,weather_reward)
add_reward(4,2,weather_reward)
add_reward(4,3,weather_reward)
add_reward(4,4,weather_reward)
add_reward(4,5,weather_reward)
add_reward(4,6,weather_reward)

add_reward(4,8,weather_reward)
add_reward(4,9,weather_reward)

add_reward(3,6,weather_reward)
add_reward(1,6,weather_reward)

# Output CSV file, removing all "impossible reward" flagged transitions
flat_world = world.flatten()
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["s", "a", "r", "sp"])
    for row in flat_world:
        if (row[2] != impossible_reward):
            writer.writerow(row)

print("done")
