import os
import sys

import torch
import wandb

from redis import Redis

from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator

from agent import get_agent
from training.aasma_body.obs import AasmaObsBuilder
from training.aasma_body.parser import AasmaAction
from training.aasma_body.reward import AasmaRewardFunction

import configparser

if __name__ == "__main__":
    # Retrieve Redis password 
    redis_password = ""
    if len(sys.argv) > 2:
        print("ERROR: Too many arguments as input. Insert only Redis Authentication Password!")
        exit()
    elif len(sys.argv) < 2:
        redis_password = input("Redis Authentication Password: ")
    else:
        redis_password = sys.argv[1]

    # Parse values from config
    config_parser = configparser.ConfigParser(allow_no_value=True)
    config_parser.read("training/config.cfg")

    # - Run settings
    wandb_key_system_variable = config_parser['Learner Configuration']['wandb_key_system_variable']
    run_name = config_parser['Learner Configuration']['run_name']
    run_project = config_parser['Learner Configuration']['run_project']
    entity = config_parser['Learner Configuration']['entity']
    run_id = config_parser['Learner Configuration']['run_id']
    config = dict()

    for key, value in config_parser['Learner Configuration config'].items():
        new_value = int(value) if (float(value)).is_integer() else float(value)
        config[key] = new_value

    # - Reddis Settings
    ip = config_parser['Redis Configuration']['ip']
    worker_counter = config_parser['Redis Configuration']['worker_counter']

    # Set WANDB to log learning procedure
    wandb.login(key=os.environ[wandb_key_system_variable])
    logger = wandb.init(name=run_name, project=run_project, entity=entity, id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    # Set Redis
    redis = Redis(host=ip, password=redis_password)
    redis.delete(worker_counter)  # Reset Workers to 0 as a new learning faze will start

    # Learning Configuration
    rollout_gen = RedisRolloutGenerator(redis,
                                        lambda: AasmaObsBuilder(4),
                                        lambda: AasmaRewardFunction(),
                                        AasmaAction,
                                        save_every=logger.config.iterations_per_save,
                                        logger=logger, clear=run_id is None,
                                        max_age=1)

    agent = get_agent(actor_lr=logger.config.actor_lr, critic_lr=logger.config.critic_lr)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        ent_coef=logger.config.ent_coef,
        logger=logger,
    )
    
    # Restart from Checkpoint
    if run_id is not None:
        alg.load("ppos/aasma_1655064509.4953504/aasma_640/checkpoint.pt")
        # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "D:\\log_directory\\"
    repo_dir = "D:\\repo_directory\\"
    
    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos", save_jit=False)
