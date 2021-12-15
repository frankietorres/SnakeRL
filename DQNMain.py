from DQNClasses import DQNLearning
from SnakeEnv import SnakeEnv
from DQNClasses import load_replay_data_from_csv
import numpy as np

# start of main

print("Creating model")
env = SnakeEnv(grid_size=10)

temp_learn = DQNLearning(env=env,
                         # target_name=str("SnakeEpisodic12Boolean10ThousandPlusDistanceReward"),
                         target_name=str("SnakeEpisodic12Boolean10Thousand"),
                         # target_name=str("SnakeEpisodicGraphTest"),
                         episode_count=10000,
                         min_batch_size=500,
                         max_batch_size=-1,
                         epsilon=0.1,
                         load_model=True,
                         fit_on_step=2,
                         train=False,
                         save_model=False,
                         show_graphs=True)

print("Training model")
temp_agent = temp_learn.train(debug=False)

score_list = temp_learn.auto_evaluate_graph(agent=temp_agent, num_of_times=1000)
print("Average Score: " + str(np.mean(score_list)))
print("Max Score: " + str(np.max(score_list)))
print("Min Score: " + str(np.min(score_list)))

avg_score = temp_learn.auto_evaluate(agent=temp_agent, num_of_times=100)
print("Average Score: " + str(avg_score))

temp_learn.evaluate(agent=temp_agent, num_of_times=10)

temp_learn.auto_evaluate_with_render(agent=temp_agent, num_of_times=100)
