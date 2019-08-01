import numpy as np
import pandas as pd


class rlalgorithm:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q1_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q2_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Double QLearning"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
        #BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
            state_action = self.q1_table.loc[observation, :] + self.q2_table.loc[observation, :]
           
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
         
            action = np.random.choice(self.actions)
        return action


    '''Update the Q(S,A) state-action value table using the latest experience
       This is a not a very good learning update 
    '''
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(str(s_))
        # with probability of 0.5
        if np.random.uniform() < 0.5:
            # Q1(S, A) = Q1(S, A) + learning_rate * (r + gamma * Q2(S', argmax_a(Q1(S', a))) - Q1(S, A))
            if s_ != 'terminal':
                # idxmax is argmax function
                a_argmax = self.q1_table.loc[s_, :].idxmax()
                q_target = r + self.gamma * self.q2_table.loc[s_, a_argmax]
            else:
                q_target = r  # next state is terminal
            self.q1_table.loc[s, a] = self.q1_table.loc[s, a] + self.lr * (q_target - self.q1_table.loc[s, a])

        else:
            # Q2(S, A) = Q2(S, A) + learning_rate * (r + gamma * Q1(S', argmax_a(Q2(S', a))) - Q2(S, A))
            if s_ != 'terminal':
                a_argmax = self.q2_table.loc[s_, :].idxmax()
                q_target = r + self.gamma * self.q1_table.loc[s_, a_argmax]
            else:
                q_target = r  # next state is terminal
            self.q2_table.loc[s, a] = self.q2_table.loc[s, a] + self.lr * (q_target - self.q2_table.loc[s, a])

        return s_, a_


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q1_table.index:
            # append new state to q table
            self.q1_table = self.q1_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q1_table.columns,
                    name=state,
                )
            )

        if state not in self.q2_table.index:
            # append new state to q table
            self.q2_table = self.q2_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q2_table.columns,
                    name=state,
                )
            )
