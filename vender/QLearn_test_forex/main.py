import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import mpl_toolkits.mplot3d
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.finance import candlestick_ohlc
from matplotlib import style
style.use("ggplot")

#------------------------------------------------------------------------------
# DEFINITION
#------------------------------------------------------------------------------
class RLMomentum():
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)["CLOSE"]
        self.data_full = pd.read_csv(datapath)
        self.ret = (self.data / self.data.shift(1) - 1)*100
        self.ret = self.data / self.data.shift(1) - 1        
        self.ret = self.ret.fillna(0)
       
        self.window_short = 20
        self.window_long = 60
        self.samples = len(self.data)
        self.states = 6
        self.actions = 3 # long, flat, short
        self.epsilon = 0.1
        self.gamma = 0.9 # discount factor
        self.mc = 100 # Monte Carlo

        self.q = np.zeros((self.states, self.states, self.actions))
        self.rewards = np.zeros((self.states, self.states, self.actions))
        self.count = np.zeros((self.states, self.states, self.actions), dtype = np.int16)
        self.isVisited = np.zeros((self.states, self.states, self.actions), dtype = np.bool)

        self.momentum = np.zeros(self.samples)
        self.actions_list = []
        self.rands_list = []
   
    def init(self):
        self.count = np.zeros((self.states, self.states, self.actions), dtype = np.int16)
        self.isVisited = np.zeros((self.states, self.states, self.actions), dtype = np.bool)
        self.actions_list = []

    def currentState(self, signal):
        signal = float(signal)
        sep = np.linspace(-1, 1, self.states-1)
        return sum(sep < signal)
        
    def selectAction(self, state_short, state_long):
        if (self.q[state_short, state_long, :]==0).sum() == self.actions:
            #if all action-values are 0
            return np.random.randint(0, self.actions)
        else:
            #Epsilon-Greedy
            rand = np.random.random(1)
            self.rands_list.append(rand)
            if rand < self.epsilon:
                return np.random.randint(0, self.actions)
            else:
                return np.argmax(self.q[state_short, state_long, :])

    def actionToPosition(self, action):
        if action == 0:
            return -1
        elif action == 1:
            return 0
        elif action == 2:
            return 1

    def updateRewards(self, reward, state_short, state_long, action):
        self.isVisited[state_short, state_long, action] = True
        self.rewards = self.rewards + reward * (self.gamma ** self.count)
        self.count = self.count + self.isVisited

    def updateQ(self, itr):
        self.q = (self.q * itr + self.rewards) / (itr + 1)

    def episode(self):
        for i in range(self.samples - 1): # len(data)
            if i <= self.window_long - 1: # 60
                self.momentum[i] = self.ret.ix[i] # np.zeros(self.samples)
            else:
                sub_short = self.momentum[i - self.window_short : i - 1]
                sub_long = self.momentum[i - self.window_long : i - 1]
               
                #state = annualized Sharpe ratio
                state_short = self.currentState( np.mean(sub_short) / np.std(sub_short) * np.sqrt(252) )
                state_long = self.currentState( np.mean(sub_long) / np.std(sub_long) * np.sqrt(252) )

                action = self.selectAction(state_short, state_long)
                self.actions_list.append(action)

                reward = self.ret.ix[i + 1] * self.actionToPosition(action)
                self.updateRewards(reward, state_short, state_long, action)
           
                self.momentum[i] = reward

    def plotPL(self):
        
        self.data_full["MOMENTUM"] = self.momentum
        self.data_full["MOMENTUM_ACC"] = self.momentum.cumsum()
        startdate = datetime.strptime("1/1/2006", "%m/%d/%Y")
        enddate = datetime.strptime("7/31/2017", "%m/%d/%Y")
        self.data_full["DATE"] = self.data_full["DATE"].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
        self.data_full["DATE2"] = self.data_full["DATE"].apply(lambda x: mdates.date2num(x.to_pydatetime()))

        df = self.data_full.copy()
        df = df[(startdate <= self.data_full["DATE"]) & (self.data_full["DATE"] <= enddate)]
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        ohlc_data = [list(x) for x in df[["DATE2","OPEN","HIGH","LOW","CLOSE"]].values]

        fig = plt.figure()
        ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=2)
        ax2 = plt.subplot2grid((4,4), (2,0), colspan=4, rowspan=2, sharex=ax1)
        candlestick_ohlc(ax1, ohlc_data, colorup="r", colordown="g")
        ax2.plot(df["DATE2"], df["MOMENTUM_ACC"])
        ax1.xaxis_date()
        ax1.get_xaxis().set_visible(False)
        
        ax2.axhline(y=0, linewidth=1, color='k', alpha=0.6, ls="--")

        plt.show()
        


    def monteCarlo(self):
        for i in range(self.mc):
            self.init() # count, isVisited
            self.episode() 
            print("episode",i,"done. cumulative return is",sum(self.momentum))
            self.updateQ(i)

            # print("episode",i,"done. The actions are", self.actions_list)
            print("episode", i, "The Mean is:", np.mean(self.rands_list), "The std is:", np.std(self.rands_list))
        
        # self.plotPL()
        
            #plt.plot(100 * (1 + self.momentum).cumprod(), label="RL-momentum "+str(i))



        #plt.plot(100 * (1 + self.ret).cumprod(), label="long-only")
        #plt.plot(100 * (1 + self.momentum).cumprod(), label="RL-momentum")
        #plt.legend(loc="best")
        #plt.show()
       
        #plot Q-value matrix
        # x = np.linspace(0,5,self.states)
        # y = np.linspace(0,5,self.states)
        # x,y = np.meshgrid(x, y)
       
        # for i in range(self.actions):
        #     if i == 0:
        #         position = "short"
        #     elif i == 1:
        #         position = "flat"
        #     elif i == 2:
        #         position =  "long"

        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')   
        #     ax.set_xlabel("state_short")
        #     ax.set_ylabel("state_long")
        #     ax.set_zlabel("Q-value")
        #     ax.set_title("Q-value for " + position + " position")
        #     #ax.view_init(90, 90)
        #     urf = ax.plot_surface(x, y, self.q[:, :, i], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #     plt.show()
#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
m = RLMomentum("./USDJPY_daily.csv")
m.monteCarlo()