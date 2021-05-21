import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pyglet
import math
from datetime import datetime
from pytz import timezone
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import random

la = timezone("CET")


class MultiChannelDCAEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self):
        bs_datas = pd.read_csv('Milano_bs.csv')
        self.bs_position = bs_datas[['lat','lon']]
        self.bs_position = self.bs_position.sort_values(by=['lat', 'lon'])
        self.bs_range = 0.0045045045 * 2 # 500m * 2
        self.list_bs_in_range = dict()
        self.traffic_data = np.load("mobile_traffic/npy_merge/merge_traffic.npy")
        self.row = 12
        self.col = 12


        self.feature = [0, 255, 85 , 170] 
        # 0 = available and no Current request,  255 = If assigned and Current request. 85 = available and no Current request, 170 = available and no Current request
        self.channels = 15
        # self.status = 2 #channel available //location
        self.current_base_station = [0,0]
        self.reward = 0
        self.timestep = 1
        self.blocktimes = 0
        self.total_timestep = 1
        self.total_blocktime = 0
        self.drop_times = 0
        self.state = None
        self.traffic_timestep = 0
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]
        self.queue = 0
        self.action_space = spaces.Discrete(self.channels)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.row ,self.col ,self.channels,), dtype=np.uint64)

        self.viewer = None
        self.seed()

        self.array_render = np.zeros([self.row, self.col], dtype=object)

        for i in range(143):
            tmp = []
            for j in range(143):
                distance = math.sqrt(pow(self.bs_position.iloc[i,0] - self.bs_position.iloc[j,0],2) + pow(self.bs_position.iloc[i,1] - self.bs_position.iloc[j,1],2))
                if distance < self.bs_range and distance > 0:
                    tmp.append(j)
            self.list_bs_in_range[i] = tmp





    def check_dca_real_bs(self, action, state):
        cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
        if cur_index >= 143:
            cur_index = 142
        neighbor = self.list_bs_in_range[cur_index]
        for i in range(len(neighbor)):
            row = i // 12
            col = i % 12
            if state[row, col, action] == 170 or state[row, col, action] == 255:
                return False
        return True

    def is_channel_avalable(self, state):
        used_channel = 0
        cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
        if cur_index >= 143:
            cur_index = 142
        neighbor = self.list_bs_in_range[cur_index]
        for i in range(len(neighbor)):
            row = i // 12
            col = i % 12
            unique, counts = np.unique(state[row,col,:], return_counts=True)
            channels_dict = dict(zip(unique, counts))
            if 255 in channels_dict:
                used_channel += channels_dict[255]

        if used_channel >= self.channels:
            return False
        return True



    def next_request(self, state):
        self.is_nexttime = False
        self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] -= 500

        if self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] < 0:
            self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] = 0
        # state[self.current_base_station[0], self.current_base_station[1], :, 1] = 0
        # queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])

        if int(self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1]) <= 0:
            cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
            self.drop_times += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
            self.total_blocktimes += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
            self.total_timestep += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
            self.bs_available.remove(cur_index)
        if len(self.bs_available) > 0:
            random_index = np.random.randint(len(self.bs_available))
            bs_random_index = self.bs_available[random_index]
            self.current_base_station[0] = bs_random_index // self.row
            self.current_base_station[1] = bs_random_index % self.col
            while not self.is_channel_avalable(state):
                # self.blocktimes += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 250
                # self.timestep += self.status_array[self.current_base_station[0], self.current_base_station[1], 0] // 250
                self.drop_times += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
                self.total_blocktimes += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
                self.total_timestep += self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] // 500
                self.traffic_data[self.traffic_timestep, self.current_base_station[0], self.current_base_station[1], 1] = 0
            #     # cur_index = (self.current_base_station[0] * self.row) + (self.current_base_station[1] % self.col)
                self.bs_available.remove(bs_random_index)
                if (len(self.bs_available) <= 0):
                    break
                random_index = np.random.randint(len(self.bs_available))
                bs_random_index = self.bs_available[random_index]
                self.current_base_station[0] = bs_random_index // self.row
                self.current_base_station[1] = bs_random_index % self.col
        # print(bs_random_index, self.bs_available, random_index)

        if (len(self.bs_available) <= 0):
            # self.blocktimes += np.sum(self.status_array[:,:,1])
            self.traffic_timestep += 1
            self.is_nexttime = True
            self.temp_blockprob = self.get_blockprob()
            self.temp_total_blockprob = self.get_total_blockprob()
            self.temp_drop_rate = self.get_drop_rate()
            self.blocktimes = 0
            self.timestep = 1
            self.total_blocktimes = 0
            self.total_timestep = 1
            self.drop_times = 0
            if self.traffic_timestep - self.temp_timestep >= 1:
                self.done = True
            self.set_timestamp()
            state = np.zeros([self.row, self.col, self.channels], dtype=np.uint64)
            
            # self.status_array = np.zeros((self.row,self.col))
            # for i in range(self.row):
            #     for j in range(self.col):
            #         self.status_array[i,j] = self.traffic_data[self.traffic_timestep, i, j, 1]
            self.bs_available = []
            for i in range(144):
                self.bs_available.append(i)
        # queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])

        for i in range(self.channels):
            if state[self.current_base_station[0], self.current_base_station[1], i] == 0:
                state[self.current_base_station[0], self.current_base_station[1], i] = 85
            elif state[self.current_base_station[0], self.current_base_station[1], i] == 255:
                state[self.current_base_station[0], self.current_base_station[1], i] = 170
        return state
        


    def step(self, action):
        # action = action
        # print(self.current_base_station)
        self.done = False
        state = self.state
        # temp = [self.current_base_station[0], self.current_base_station[1]]
        for i in range(self.channels):
            if state[self.current_base_station[0], self.current_base_station[1], i] == 85:
                state[self.current_base_station[0], self.current_base_station[1], i] = 0
            elif state[self.current_base_station[0], self.current_base_station[1], i] == 170:
                state[self.current_base_station[0], self.current_base_station[1], i] = 255
        if self.check_dca_real_bs(action, state):
            self.reward = 1
            # queue = int(self.status_array[self.current_base_station[0], self.current_base_station[1], 1])
            state[self.current_base_station[0], self.current_base_station[1], action] = 255
            state = self.next_request(state)
        else:
            self.blocktimes += 1
            self.total_blocktimes += 1
            self.reward = -1
            state = self.next_request(state)
        self.state = state
        self.timestep +=1
        self.total_timestep +=1
        # print("current = ", temp[0], temp[1], state[temp[0], temp[1], :])
        # print("next = ", self.current_base_station[0], self.current_base_station[1], state[self.current_base_station[0], self.current_base_station[1], :])
        return np.reshape(self.state, self.observation_space.shape), self.reward, self.done, {'timestamp' : self.get_timestamp(), 'is_nexttime' : self.is_nexttime, 'temp_blockprob' : self.temp_blockprob, 'temp_total_blockprob' : self.temp_total_blockprob, 'drop_rate' : self.temp_drop_rate}

    def get_timestamp(self):
        return datetime.fromtimestamp(self.timestamp, la).ctime()

    def get_blockprob(self):
        return self.blocktimes/self.timestep

    def get_total_blockprob(self):
        return self.total_blocktimes/self.total_timestep

    def get_drop_rate(self):
        return self.drop_times/self.total_timestep

    def set_timestamp(self):
        self.timestamp = self.traffic_data[ self.traffic_timestep, 0, 0, 0]

    def reset(self):
        self.drop_times = 0
        self.temp_drop_rate = 0
        self.temp_blockprob = 0
        self.temp_total_blockprob = 0
        self.is_nexttime = False
        self.timestep = 1
        self.blocktimes = 0
        self.total_timestep = 1
        self.total_blocktimes = 0
        self.current_base_station = [0,0]
        state = np.zeros([self.row, self.col, self.channels], dtype=np.uint64)
        if self.traffic_timestep >= 8630: #8630
            self.traffic_timestep = 0
        self.temp_timestep = self.traffic_timestep
        self.status_array = np.zeros((self.row,self.col))
        # for i in range(self.row):
        #     for j in range(self.col):
        #         self.status_array[i,j] = self.traffic_data[self.traffic_timestep, i, j, 1]
        self.bs_available = []
        for i in range(144):
            self.bs_available.append(i)


        random_index = np.random.randint(len(self.bs_available))
        bs_random_index = self.bs_available[random_index]
        self.current_base_station[0] = bs_random_index // self.row
        self.current_base_station[1] = bs_random_index % self.col
        state[self.current_base_station[0], self.current_base_station[1], :] = 85
        self.state = state
        return np.reshape(self.state, self.observation_space.shape)

    def render(self, mode='human'):
        class DrawText:
            def __init__(self, label:pyglet.text.Label):	
                self.label=label	
            def render(self):	
                self.label.draw()
        screen_width = 800
        screen_height = 600
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            x=30
            y=screen_height-10
            for i in range(self.row):
                x = 30 + i * 20
                for j in range(self.col):
                    bs = rendering.make_polygon([(x,y),(x-20,y-13),(x-20,y-40),(x-0,y-53),(x+20,y-40),(x+20,y-13),(x-0,y-0)], False)
                    label = pyglet.text.Label(str(int(self.channels - np.sum(self.state[i,j,:,0])/255)),
                                    font_size=10,
                                    x=x-5, y=y-25,
                                    anchor_x='left', anchor_y='center', color=(255, 0, 0, 255))
                    self.array_render[i,j] = label
                    self.viewer.add_geom(DrawText(label))
                    x = x + 40
                    self.viewer.add_geom(bs)
                y = y - 40
            self.timestamp_label = pyglet.text.Label(str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S')),
                font_size=10,
                x=screen_width-150, y=screen_height - 10,
                anchor_x='left', anchor_y='center', color=(255,0,255, 255))
            self.viewer.add_geom(DrawText(self.timestamp_label))
        else:
            for i in range(self.row):
                for j in range(self.col):
                    self.array_render[i,j].text = str(int(self.channels - np.sum(self.state[i,j,:,0])/255))
            self.timestamp_label.text = str(datetime.fromtimestamp(self.timestamp, la).strftime('%Y-%m-%d %H:%M:%S'))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



        

