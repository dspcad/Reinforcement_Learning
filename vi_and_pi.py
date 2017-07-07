### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
	V_prev = np.zeros(nS)
        for itr in xrange(max_iteration):
          for st in range(nS):

            for a in range(4):
              gamma_sum = 0.0
              next_st = P[st][a][0][1]
              gamma_sum += gamma*P[st][a][0][0]*V[next_st]         

              #print 'state: ', st
              #print 'gama sum: ', gamma_sum
              #print 'reward: ', P[st][policy[st]][0][2]
              Q = P[st][a][0][2] + gamma_sum

              if Q > V[st]:
                #print 'gama sum: ', gamma_sum
                #print 'Q: ', Q
                V_prev[st] = V[st]
                V[st] = Q




          if np.all(np.abs(V-V_prev) < tol):
            #print "iteration: ", itr
            break


	############################
	#    Extract the policy    #
	############################
        for st in range(nS):

          max_gamma_sum = 0.0
          argmax_a = 0
          for a in range(4):
            gamma_sum = 0.0
            next_st = P[st][a][0][1]
            gamma_sum += gamma*P[st][a][0][0]*V[next_st]         

            #print 'state: ', st
            #print 'gama sum: ', gamma_sum
            #print 'reward: ', P[st][policy[st]][0][2]
            Q = P[st][a][0][2] + gamma_sum

            if Q > max_gamma_sum:
              #print 'gama sum: ', gamma_sum
              #print 'Q: ', Q
              max_gamma_sum = Q
              argmax_a = a

          policy[st] = argmax_a

	return V, policy


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""

	V      = np.zeros(nS)
	V_prev = np.zeros(nS)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        for itr in xrange(max_iteration):
          for st in range(nS):
            gamma_sum = 0.0
            next_st = P[st][policy[st]][0][1]
            #print st, next_st
            gamma_sum += gamma*P[st][policy[st]][0][0]*V[next_st]         

            #print 'state: ', st
            #print 'gama sum: ', gamma_sum
            #print 'reward: ', P[st][policy[st]][0][2]
            V_prev[st] = V[st]
            V[st]      = P[st][policy[st]][0][2] + gamma_sum

            #print "value function: ", V

          if np.all(np.abs(V-V_prev) < tol):
            #print "iteration: ", itr
            break

	return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""    
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        for st in range(nS):

          max_gamma_sum = 0.0
          argmax_a = 0
          for a in range(4):
            gamma_sum = 0.0
            next_st = P[st][a][0][1]
            gamma_sum += gamma*P[st][a][0][0]*value_from_policy[next_st]         

            #print 'state: ', st
            #print 'gama sum: ', gamma_sum
            #print 'reward: ', P[st][policy[st]][0][2]
            Q = P[st][a][0][2] + gamma_sum

            if Q > max_gamma_sum:
              #print 'gama sum: ', gamma_sum
              #print 'Q: ', Q
              max_gamma_sum = Q
              argmax_a = a

          policy[st] = argmax_a
      

	return policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""

        #
        #             3
        #             ^ 
        #             |
        #        0 <-   -> 2
        #             |
        #             1
        #

	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	############################
        for itr in range(max_iteration):
          V = policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3)
          print "value function: ", V
          policy = policy_improvement(P, nS, nA, V, policy, gamma=0.9)
          print "policy: ", policy
  
	return V, policy



def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0); 
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print "Episode reward: %f" % episode_reward


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.

if __name__ == "__main__":
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print env.__doc__
	print "Here is an example of state, action, reward, and next state"
	#example(env)
	#print env.P.keys()
	#print env.P.values()
        for i in range(env.nS):
	  print "state ", i, " : ", env.P[i]
        print env.nS
        print env.nA
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=2000, tol=1e-3)
	render_single(env, p_vi)
	#V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	#render_single(env, p_pi)
