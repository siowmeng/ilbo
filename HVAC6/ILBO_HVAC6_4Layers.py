#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import scipy.stats
import math
import matplotlib.pyplot as plt
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

tf.enable_eager_execution()
tf.keras.backend.set_floatx('float32')

# Gaussian Noise Class (Diagonal Covariance Matrix)
class GaussActionNoise:
    def __init__(self, mean, std_deviation, dim = 2):
        self.mean = mean
        self.std_dev = std_deviation
        self.dim = dim

    def __call__(self):
        x = np.random.normal(self.mean, self.std_dev, self.dim)
        return x

# Parent Buffer class
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, prioritized_replay_eps=1e-6):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.prioritized_replay_eps = prioritized_replay_eps
    
    def __len__(self):
        return len(self.buffer)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs, action, rew, new_obs, done):
        self.buffer.add(obs, action, rew, new_obs, float(done))
    
    def learn(self, beta):
        
        experience = self.buffer.sample(self.batch_size, beta = beta)
        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
        
        rewards = tf.expand_dims(rewards, 1)
        
        update_metrics = self.update(obses_t, actions, rewards, obses_tp1, dones, weights.astype(np.float32))
        
        td_errors = update_metrics[0]
        
        # update priorities
        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                
        self.buffer.update_priorities(batch_idxes, new_priorities)
        
        return update_metrics

# Q(s,a) Buffer
class QsaBuffer(Buffer):
    
    def __init__(self, buffer_capacity=100000, batch_size=64, alpha = 0.6):
        
        super(QsaBuffer, self).__init__(buffer_capacity, batch_size, prioritized_replay_eps = 1e-6)
        
        self.buffer = PrioritizedReplayBuffer(self.buffer_capacity, alpha = alpha)
    
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch, impt_weights_batch
    ):
        with tf.GradientTape() as tape:
            target_actions = mm_target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * mm_target_qsa(
                [next_state_batch, target_actions], training=True
            )
            qsa_value = mm_qsa([state_batch, action_batch], training=True)
            td_errors = y - qsa_value
            qsa_loss = tf.math.reduce_mean(impt_weights_batch * tf.math.square(td_errors))
        
        qsa_grad = tape.gradient(qsa_loss, mm_qsa.trainable_variables)
        mm_qsa_optimizer.apply_gradients(
            zip(qsa_grad, mm_qsa.trainable_variables)
        )
        
        qsa_grad_list = []
        for grad in qsa_grad:
            qsa_grad_list.append(tf.math.reduce_mean(tf.abs(grad)))
        
        return td_errors, qsa_loss, tf.math.reduce_mean(qsa_grad_list)

# MM Actor Buffer Class
class ActorBuffer(Buffer):
    
    def __init__(self, buffer_capacity=100000, batch_size=64):
        
        super(ActorBuffer, self).__init__(buffer_capacity, batch_size)
        
        self.buffer = ReplayBuffer(self.buffer_capacity)
    
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, 
    ):
        with tf.GradientTape() as tape:
            
            actions = mm_actor(state_batch, training = True)
            
            nextState_Mean = state_batch \
                                + time_delta / cap * (actions * cap_air * (temp_air - state_batch)) * is_room \
                                + (tf.linalg.matmul(state_batch, tf.cast(adj_matrix / r_wall_matrix, tf.float32)) \
                                   - tf.math.multiply(state_batch, tf.cast(tf.math.reduce_sum(adj_matrix / r_wall_matrix, 1), tf.float32))) \
                                + adj_out * temp_out_mean / r_outside \
                                - adj_out * state_batch / r_outside \
                                + adj_hall * temp_hall_mean / r_hall \
                                - adj_hall * state_batch / r_hall
                        
            ndim = state_batch.get_shape().as_list()[1]
            diffMatrix = next_state_batch - nextState_Mean
            
            prob_nextState = tf.math.exp(-0.5 * tf.reduce_sum(tf.matmul(diffMatrix, tf.cast(tf.linalg.inv(gaussian_cov), tf.float32)) * diffMatrix, 1)) / tf.math.sqrt((2 * math.pi)**ndim * tf.cast(tf.linalg.det(gaussian_cov), tf.float32))
            
            prob_nextState /= max_pdf # Divide by maximum possible pdf (normalize to below 1)
            prob_nextState += 1e-12 # Small value tolerance (avoid pdf being 0 due to limited accuracy)
            
            #next_state_actions = tf.dtypes.cast(actor_model(next_state_batch, training = True), tf.float64)
            #next_state_actions = tf.dtypes.cast(target_actor(next_state_batch, training = True), tf.float64)
            V_currState = mm_qsa([state_batch, actions], training=True)
            next_state_actions = mm_lag_actor(next_state_batch, training = True)
            V_nextState = mm_qsa([next_state_batch, next_state_actions], training=True)
            
            out_of_range_bool = (state_batch < temp_low_vec) | (state_batch > temp_up_vec)
            reward_sa = -tf.math.reduce_sum(is_room * (actions * cost_air_var \
                                                       + tf.cast(out_of_range_bool, tf.float32) * (penalty_var) \
                                                       + 10.0 * tf.math.abs((temp_up_vec + temp_low_vec) / 2.0 - state_batch)), 
                                            axis = 1)
            
            #actor_loss = -tf.math.reduce_mean(gamma * V_nextState * tf.math.log(prob_nextState))
            
            actor_loss = -tf.math.reduce_mean(reward_sa + gamma * (V_nextState - V_currState) * tf.math.log(prob_nextState))
            
        actor_grad = tape.gradient(actor_loss, mm_actor.trainable_variables)
        mm_actor_optimizer.apply_gradients(
            zip(actor_grad, mm_actor.trainable_variables)
        )
        
        actor_grad_list = []
        for grad in actor_grad:
            actor_grad_list.append(tf.math.reduce_mean(tf.abs(grad)))
                
        return actor_loss, tf.math.reduce_mean(actor_grad_list)
    
    # For Actor buffer
    def learn(self):
                
        obses_t, actions, rewards, obses_tp1, dones = self.buffer.sample(self.batch_size)
        
        rewards = tf.expand_dims(rewards, 1)
                
        update_metrics = self.update(obses_t, actions, rewards, obses_tp1)
        
        return update_metrics

# Actor Network Architecture
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation = "relu")(inputs)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(128, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(64, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(32, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(num_actions, activation="sigmoid", 
                           kernel_initializer=last_init)(out)
    
    outputs = outputs * air_max_vec
    model = tf.keras.Model(inputs, outputs)
    return model

# Qsa Network Architecture
def get_qsa():
    
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(32, activation = "relu")(state_input)
    state_out = layers.LayerNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation = "relu")(action_input)
    action_out = layers.LayerNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    
    out = layers.Dense(256, activation = "relu")(concat)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(128, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(64, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(32, activation = "relu")(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(1, 
                           activation="relu", 
                           kernel_initializer=last_init)(out)
    
    # Try RELU output to make it +ve (QMix)    
    outputs = outputs * -1.0
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

# Policy Function
def policy(actor_model, state, noise_object, t):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + (noise * air_max_vec)

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, 0, air_max_vec)

    return np.squeeze(legal_action)

# Calculate Cumulative Discounted Rewards
def calcDiscRewards(traj_rewards, gamma):
    i, total_reward = 0, 0
    for r in traj_rewards:
        total_reward += ((gamma**i) * r)
        i += 1
    return total_reward

# This updates target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

@tf.function
def update_lag_actor(lag_actor_weights, weights):
    for (a, b) in zip(lag_actor_weights, weights):
        a.assign(b)

def set_start_state(rddl_env, start_state):
    
    # Initialize Environment
    state, t = rddl_env.reset()
    
    # Set Start State
    state['temp/1'] = np.array(start_state)
    env._state['temp/1'] = state['temp/1']
    
    return rddl_env, state, t

def simulate_test(rddl_env, start_state, total_episodes, seed):
    # list of trajectories
    listTrajectory = []
    listTestTrajectories = []
    disc_rewards_arr, mean_qsa_loss_arr, mean_actor_loss_arr, mean_qsa_grad_arr, mean_actor_grad_arr = [], [], [], [], []
    
    beta_schedule = LinearSchedule(total_episodes * horizon, 
                                   initial_p = 1.0, 
                                   final_p = 1.0)
    
    noise_schedule = LinearSchedule(total_episodes * horizon, 
                                    initial_p = 0.2, 
                                    final_p = 0.0)
    
    t_iter, best_mean_undisc_reward = 0, float('-inf')
    
    for ep in range(total_episodes):
    
        # initialize environament
        if ep == 0:
            rddl_env, state, t = set_start_state(rddl_env, start_state)
        else:
            rddl_env, state, t = set_start_state(rddl_env, np.random.uniform([0] * num_states, [30] * num_states))
        done = False
        
        # create a trajectory container
        trajectory = rddlgym.Trajectory(rddl_env)
        qsa_loss_arr, actor_loss_arr, qsa_grad_arr, actor_grad_arr = [], [], [], []
    
        while not done:
            
            curr_state = state['temp/1'].astype(np.float32)            
            
            tf_state = tf.expand_dims(tf.convert_to_tensor(curr_state), 0)
            
            action = OrderedDict({'air/1': policy(mm_actor, 
                                                  tf_state, 
                                                  GaussActionNoise(mean = 0, 
                                                                   std_deviation = noise_schedule.value(t_iter), 
                                                                   dim = num_actions), 
                                                  t)})
            
            next_state, reward, done, info = rddl_env.step(action)            
            reward, nextState = reward.astype(np.float32), next_state['temp/1'].astype(np.float32)
            
            # Reward scaling for HVAC-6, for training only
            scaled_reward = reward / np.abs(penalty_var) * 10.0
            scaled_reward = scaled_reward.astype(np.float32)
            
            q_buffer.record(curr_state, action['air/1'].astype(np.float32), scaled_reward, nextState, done)
            actor_buffer.record(curr_state, action['air/1'].astype(np.float32), scaled_reward, nextState, done)
            if len(q_buffer) > q_buffer.batch_size:
                td_errors, qsa_loss, ave_qsa_grad = q_buffer.learn(beta = beta_schedule.value(t_iter))
                qsa_loss_arr.append(qsa_loss)
                qsa_grad_arr.append(ave_qsa_grad)
                update_target(mm_target_qsa.variables, mm_qsa.variables, tau)
            if (len(actor_buffer) > actor_buffer.batch_size):
                actor_loss, ave_actor_grad = actor_buffer.learn()
                actor_loss_arr.append(actor_loss)
                actor_grad_arr.append(ave_actor_grad)
                update_target(mm_target_actor.variables, mm_actor.variables, tau)
                update_lag_actor(mm_lag_actor.variables, mm_actor.variables)
                
            trajectory.add_transition(t, state, action, reward, next_state, info, done)
    
            state = next_state
            t = rddl_env.timestep
            
            t_iter += 1
        
        disc_rewards = calcDiscRewards(trajectory.as_dataframe().reward, gamma)
        disc_rewards_arr.append(disc_rewards)
        
        if len(qsa_loss_arr) == 0:
            mean_qsa_loss = None
            mean_qsa_loss_arr.append(float('nan'))
        else:
            mean_qsa_loss = np.mean(qsa_loss_arr)
            mean_qsa_loss_arr.append(mean_qsa_loss)
        
        if len(actor_loss_arr) == 0:
            mean_actor_loss = None
            mean_actor_loss_arr.append(float('nan'))
        else:
            mean_actor_loss = np.mean(actor_loss_arr)
            mean_actor_loss_arr.append(mean_actor_loss)
        
        if len(qsa_grad_arr) == 0:
            mean_qsa_grad = None
            mean_qsa_grad_arr.append(float('nan'))
        else:
            mean_qsa_grad = np.mean(qsa_grad_arr)
            mean_qsa_grad_arr.append(mean_qsa_grad)
        
        if len(actor_grad_arr) == 0:
            mean_actor_grad = None
            mean_actor_grad_arr.append(float('nan'))
        else:
            mean_actor_grad = np.mean(actor_grad_arr)
            mean_actor_grad_arr.append(mean_actor_grad)
        
        print("Episode * {} * Total Reward is ==> {}".format(ep, disc_rewards))
        print("Qsa loss: {}".format(mean_qsa_loss))
        print("Actor loss: {}".format(mean_actor_loss))
        print("Average Qsa gradient: {}".format(mean_qsa_grad))
        print("Average actor gradient: {}".format(mean_actor_grad))
        print()
        
        listTrajectory.append(trajectory.as_dataframe())
        
        if (ep + 1) % test_interval == 0:
            
            l_test_trajs, mean_disc_r, mean_undisc_r = test_actor_loop(folderName + '/' + 'mm_test_log_' + str(ep + 1) + '.csv', env, start_state)
            listTestTrajectories.append(l_test_trajs)
            
            if mean_undisc_r > best_mean_undisc_reward:
                
                best_mm_actor.set_weights(mm_actor.get_weights())
                best_mm_qsa.set_weights(mm_qsa.get_weights())
                
                best_mean_undisc_reward = mean_undisc_r
        
    return disc_rewards_arr, mean_qsa_loss_arr, mean_actor_loss_arr, mean_qsa_grad_arr, mean_actor_grad_arr, listTrajectory, listTestTrajectories

def test_actor_loop(filename, rddl_env, start_state):
    
    list_traj_df, list_disc_reward, list_undisc_reward = [], [], []
    
    for i in range(test_loops):
        
        # initialize environament
        rddl_env, state, t = set_start_state(rddl_env, start_state)
        done = False

        test_trajectory = rddlgym.Trajectory(rddl_env)

        while not done:
            
            curr_state = state['temp/1'].astype(np.float32)

            tf_state = tf.expand_dims(tf.convert_to_tensor(curr_state), 0)
            
            action = OrderedDict({'air/1': policy(mm_actor, tf_state, lambda : np.array([0] * num_actions), t)})
            next_state, reward, done, info = rddl_env.step(action)

            test_trajectory.add_transition(t, state, action, reward, next_state, info, done)

            state = next_state
            t = rddl_env.timestep

        test_log_df = test_trajectory.as_dataframe()
        disc_reward = calcDiscRewards(test_log_df.reward, gamma)
        test_log_df['Total Discounted Rewards'] = [disc_reward for i in range(test_log_df.shape[0])]
        undisc_reward = calcDiscRewards(test_log_df.reward, 1.0)
        test_log_df['Total Undiscounted Rewards'] = [undisc_reward for i in range(test_log_df.shape[0])]
        
        list_traj_df.append(test_log_df)
        list_disc_reward.append(disc_reward)
        list_undisc_reward.append(undisc_reward)
    
    return list_traj_df, np.mean(list_disc_reward), np.mean(list_undisc_reward)

def log_learn(folderName, lDiscRewards, lQsaLoss, lActorLoss, lQsaGrad, lActorGrad):
    
    learn_log_df = pd.DataFrame({'Episode': [i for i in range(len(lDiscRewards))], 
                                 'Discounted Rewards': lDiscRewards, 
                                 'Qsa Loss': lQsaLoss, 
                                 'Actor Loss': lActorLoss, 
                                 'Qsa Gradient': lQsaGrad, 
                                 'Actor Gradient': lActorGrad})
    
    learn_log_df.to_csv(folderName + 'learn_log.csv', index = False)

def log_trajectories(folderName, lTrainTraj, lListTestTraj):
    
    for i in range(len(lTrainTraj)):
        lTrainTraj[i].to_csv(folderName + 'E' + str(i + 1) + '.csv', index = False)
    
    testTrajFolder = folderName + 'test_trajs/'
    for i in range(len(lListTestTraj)):
        testTraj_subFolder = testTrajFolder + 'E' + str((i + 1) * test_interval) + '/'
        pathlib.Path(testTraj_subFolder).mkdir(parents = True, exist_ok = True)
        for j in range(len(lListTestTraj[i])):
            lListTestTraj[i][j].to_csv(testTraj_subFolder + str(j + 1) + '.csv', index = False)

def plot_graphs(dirName, lQsaLoss, lActorLoss, lQsaGrad, lActorGrad, lTrajList):
    
    numEpisodes = len(lQsaLoss)
    plt.figure()
    plt.plot(range(numEpisodes), lQsaLoss)
    plt.xlabel("Episode")
    plt.ylabel("Average Qsa Loss Across Minibatches")
    plt.savefig(dirName + 'qsa_loss.png')
    plt.close()
    
    # Plot mean_actor_loss_arr
    plt.figure()
    plt.plot(range(numEpisodes), lActorLoss)
    plt.xlabel("Episode")
    plt.ylabel("Average Actor Loss Across Minibatches")
    plt.savefig(dirName + 'actor_loss.png')
    plt.close()
    
    # Plot mean_qsa_grad_arr
    plt.figure()
    plt.plot(range(numEpisodes), lQsaGrad)
    plt.xlabel("Episode")
    plt.ylabel("Average Qsa Gradient Across Minibatches")
    plt.savefig(dirName + 'qsa_grad.png')
    plt.close()
    
    # Plot mean_qsa_loss_arr, mean_actor_loss_arr
    plt.figure()
    plt.plot(range(numEpisodes), lActorGrad)
    plt.xlabel("Episode")
    plt.ylabel("Average Actor Gradient Across Minibatches")
    plt.savefig(dirName + 'actor_grad.png')
    plt.close()
    
    ave_disc_rewards = []
    ave_undisc_rewards = []
    for TrajList in lTrajList:
        ave_disc_rewards.append(np.mean([df['Total Discounted Rewards'][0] for df in TrajList]))
        ave_undisc_rewards.append(np.mean([df['Total Undiscounted Rewards'][0] for df in TrajList]))
    
    testEpisode_range = range(test_interval, len(ave_undisc_rewards) * test_interval + 1, test_interval)
    # Plot ave_disc_rewards, ave_undisc_rewards
    plt.figure()
    plt.plot(testEpisode_range, ave_disc_rewards)
    plt.xlabel("Sample Episodes Collected")
    plt.ylabel("Accumulated Discounted Rewards")
    plt.savefig(dirName + 'test_disc_rewards.png')
    #plt.show()
    plt.close()
    
    # Plot ave_disc_rewards, ave_undisc_rewards
    plt.figure()
    plt.plot(testEpisode_range, ave_undisc_rewards)
    plt.xlabel("Sample Episodes Collected")
    plt.ylabel("Accumulated Undiscounted Rewards")
    plt.savefig(dirName + 'test_undisc_rewards.png')
    #plt.show()
    plt.close()
    
    ave_reward_df = pd.DataFrame(data = {'Episode': testEpisode_range, 
                                         'Average Discounted Reward': ave_disc_rewards,
                                         'Average Undiscounted Reward': ave_undisc_rewards})
    
    ave_reward_df.to_csv(dirName + 'average_test_reward.csv', index = False)

def init_networks(qsa_lr, actor_lr):
    
    actor_model = get_actor()
    qsa_model = get_qsa()
    target_actor = get_actor()
    target_qsa = get_qsa()
    
    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_qsa.set_weights(qsa_model.get_weights())
    
    qsa_optimizer = tf.keras.optimizers.RMSprop(qsa_lr)
    actor_optimizer = tf.keras.optimizers.RMSprop(actor_lr)
    
    lag_actor = get_actor()
    lag_actor.set_weights(actor_model.get_weights())
    return actor_model, qsa_model, target_actor, target_qsa, lag_actor, qsa_optimizer, actor_optimizer

# Start States
with open('./start_state.txt') as f:
    startState_list = [tuple(map(float, i.split(','))) for i in f]

# Seeds
with open('./seed.txt') as f:
    seed_list = [int(i) for i in f]

import rddlgym

# create RDDLGYM environment
rddl_id = "HVAC-6" # see available RDDL domains/instances with `rddlgym ls` command
env = rddlgym.make(rddl_id, mode = rddlgym.GYM)

# you can also wrap your own RDDL files (domain + instance)
# env = rddlgym.make("/path/to/your/domain_instance.rddl", mode=rddlgym.GYM)

num_states = env.observation_space['temp/1'].shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space['air/1'].shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space['air/1'].high[0]
lower_bound = env.action_space['air/1'].low[0]

#upper_bound = 1.0
#lower_bound = -1.0

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

adj_matrix = env.non_fluents['ADJ/2']

for i in range(adj_matrix.shape[0]):
    for j in range(i + 1, adj_matrix.shape[0]):
        adj_matrix[j, i] = adj_matrix[i, j]

#adj_out = [1, 0, 1, 1, 0, 1]
#adj_hall = [1, 1, 1, 1, 1, 1]
adj_out = env.non_fluents['ADJ_OUTSIDE/1']
adj_hall = env.non_fluents['ADJ_HALL/1']
time_delta = env.non_fluents['TIME_DELTA/0']
cap = env.non_fluents['CAP/1']
cap_air = env.non_fluents['CAP_AIR/0']
is_room = env.non_fluents['IS_ROOM/1']
r_wall_matrix = env.non_fluents['R_WALL/2']
r_outside = env.non_fluents['R_OUTSIDE/1']
r_hall = env.non_fluents['R_HALL/1']
temp_air = env.non_fluents['TEMP_AIR/0']
temp_out_mean = env.non_fluents['TEMP_OUTSIDE_MEAN/1']
temp_out_var = env.non_fluents['TEMP_OUTSIDE_VARIANCE/1']
temp_hall_mean = env.non_fluents['TEMP_HALL_MEAN/1']
temp_hall_var = env.non_fluents['TEMP_HALL_VARIANCE/1']
cost_air_var = env.non_fluents['COST_AIR/0']
temp_up_vec = env.non_fluents['TEMP_UP/1']
temp_low_vec = env.non_fluents['TEMP_LOW/1']
penalty_var = env.non_fluents['PENALTY/0']
air_max_vec = env.non_fluents['AIR_MAX/1']
horizon = env.horizon

#gaussian_sd = 1e-1
#gaussian_sd = 0.05**0.5
gaussian_cov = np.diag(adj_out * temp_out_var / r_outside**2 + adj_hall * temp_hall_var / r_hall**2)
gaussian_zero_mean = np.array([0] * num_states)
max_pdf = scipy.stats.multivariate_normal.pdf(gaussian_zero_mean, gaussian_zero_mean, gaussian_cov)

# Learning rate for Qsa and DRP
mm_qsa_lr = 0.001
mm_actor_lr = 0.0001

total_episodes = 5000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

test_interval = 5
test_loops = 64

for startState_idx in range(len(startState_list)):
        
    for seed in seed_list:
        
        folderName = './' + rddl_id + '_Results/MM/SIdx' + str(startState_idx) + '/Seed' + str(seed) + '/'
        pathlib.Path(folderName).mkdir(parents = True, exist_ok = True)
        
        # Initialize for MM
        tf.set_random_seed(seed)
        mm_actor, mm_qsa, mm_target_actor, mm_target_qsa, mm_lag_actor, mm_qsa_optimizer, mm_actor_optimizer = init_networks(mm_qsa_lr, mm_actor_lr)
        best_mm_qsa = get_qsa()
        best_mm_qsa.set_weights(mm_qsa.get_weights())
        best_mm_actor = get_actor()
        best_mm_actor.set_weights(mm_actor.get_weights())
        q_buffer = QsaBuffer(1000000, 64, alpha = 0.3)
        actor_buffer = ActorBuffer(1000, 64)
        
        print("MM")
        # MM Second
        discRewards_l, qsaLoss_l, actorLoss_l, qsaGrad_l, actorGrad_l, MMTrainTraj_l, listMMTestTraj_l = simulate_test(env, startState_list[startState_idx], total_episodes, seed)
        
        mm_qsa.save_weights(folderName + 'checkpoints/mm_qsa')
        mm_actor.save_weights(folderName + 'checkpoints/mm_actor')
        best_mm_qsa.save_weights(folderName + 'checkpoints/best_mm_qsa')
        best_mm_actor.save_weights(folderName + 'checkpoints/best_mm_actor')
        
        log_learn(folderName, discRewards_l, qsaLoss_l, actorLoss_l, qsaGrad_l, actorGrad_l)
        log_trajectories(folderName, MMTrainTraj_l, listMMTestTraj_l)
        
        plot_graphs(folderName, qsaLoss_l, actorLoss_l, qsaGrad_l, actorGrad_l, listMMTestTraj_l)
        
