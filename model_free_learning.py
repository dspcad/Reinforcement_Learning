### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  #    0
  # 1     3
  #    2
  Q = np.zeros((env.nS, env.nA))
  for i in xrange(num_episodes):
  #for i in xrange(1):
    for st in xrange(env.nS):

      #print '-------   Record   -----------'
      #print 'st:', st

      if np.random.uniform(0,1) <= e:
        a = env.action_space.sample()
      else:
        a = np.argmax(Q[st,:])
 

      #print 'action:', a
      a_sel = np.random.randint(len(env.P[st][a]))
      next_st   = env.P[st][a][a_sel][1]
      next_st_a = np.argmax(Q[next_st,:])
      r         = env.P[st][a][a_sel][2]
      Q_sample  = gamma*Q[next_st][next_st_a]
      Q_sample += r

      #Q_sample = 0.0
      #for a_sel in range(len(env.P[st][a])):
      #  next_st = env.P[st][a][a_sel][1]
      #  next_st_a = np.argmax(Q[next_st,:])
      #  r       = env.P[st][a][a_sel][2]

      #  Q_sample += gamma*env.P[st][a][a_sel][0]*Q[next_st][next_st_a]
      #  Q_sample += r
       
        #if st == 14:
        #  print "a: ", a    
        #  print "a_sel: ", a    
        #  print "r: ",  env.P[st][a][a_sel]
        #  print "r: ",  env.P[st][a][a_sel][2]
      #print 'next st:', next_st
      #print 'reward:', r
      #print '------------------------------'

      #next_st_a = np.argmax(Q[next_st,:])
      #Q_sample = Q[next_st][next_st_a]

      #Q_sample *= gamma 
      #Q_sample += r 
      Q[st][a] = (1-lr)*Q[st][a] + lr*Q_sample


    e *= decay_rate

  print Q
 
  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  Q = np.zeros((env.nS, env.nA))
  for i in xrange(num_episodes):
  #for i in xrange(1):
    for st in xrange(env.nS):

      #print '-------   Record   -----------'
      #print 'st:', st
      #if np.random.uniform(0,1) <= e:
      #  a = env.action_space.sample()
      #else:
      #  a = np.argmax(Q[st,:])
 

      a = env.action_space.sample()

      a_sel = np.random.randint(len(env.P[st][a]))
      next_st   = env.P[st][a][a_sel][1]
      next_st_a = np.argmax(Q[next_st,:])
      r         = env.P[st][a][a_sel][2]
      Q_sample  = gamma*Q[next_st][next_st_a]
      Q_sample += r


      next_st_a_sel = np.random.randint(len(env.P[st][next_st_a]))
      next_r        = env.P[st][a][next_st_a_sel][2]
      Q_sample += next_r


      #print env.P[st][a][a_sel]
      #print "a: ", a    
      #print "a_sel: ", a    
      #print "r: ",  env.P[st][a][a_sel]
      #print "r: ",  env.P[st][a][a_sel][2]
   

      #print "Q_sample_random: ", Q_sample_random
      #print "Q_sample: ", Q_sample

      Q[st][a] = (1-lr)*Q[st][a] + lr*Q_sample


    e *= decay_rate

  print Q
 
  return Q


  return np.ones((env.nS, env.nA))

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """
  episode_reward = 0
  for i in xrange(100):
    state = env.reset()
    done = False
    while not done:
      env.render()
      time.sleep(0.5) # Seconds between frames. Modify as you wish.
      #print "state:  ", state
      action = np.argmax(Q[state])
      #print "action: ", action
      #print "next state: ", env.P[state][action]
      state, reward, done, _ = env.step(action)
      episode_reward += reward

    print "Iteration %d      Episode reward: %f" % (i,episode_reward)

  print "Episode reward: %f" % episode_reward
# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  #Q = learn_Q_QLearning(env)
  Q = learn_Q_SARSA(env)

  #for i in range(env.nS):
  #  print "state ", i, " : ", env.P[i][0]
  #  print "state ", i, " : ", env.P[i][1]
  #  print "state ", i, " : ", env.P[i][2]
  #  print "state ", i, " : ", env.P[i][3]
  render_single_Q(env, Q)

if __name__ == '__main__':
    main()

