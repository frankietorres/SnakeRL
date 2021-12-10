from DQNClasses import DQNLearning
from SnakeEnv import SnakeEnv
from DQNClasses import load_replay_data_from_csv
# start of main

print("Creating model")
env = SnakeEnv(grid_size=10)

temp_learn = DQNLearning(env=env,
                         target_name=str("SnakeEpisodic12Boolean10Thousand"),
                         episode_count=100000,
                         min_batch_size=500,
                         max_batch_size=-1,
                         epsilon=0.1,
                         load_model=True,
                         fit_on_step=2,
                         train=False,
                         save_model=False,
                         show_graphs=False)

print("Training model")
temp_agent = temp_learn.train(debug=False)

temp_learn.evaluate(agent=temp_agent,
                    num_of_times=10)
