## Aasma Rocket League 2vs2 Agent 

### Description

The project focus on designing and implementing a multiagent system while considering cooperation,
coordination, or negotiation strategies.
The project includes several of the following features:
• Agents have either conflicting goals or face complex coordination problems;
• Agents have a variety of sensors and actuators (and they are not be too limited);
• Communication and coordination mechanisms (cooperation, teamwork);
• Complex negotiation and cooperation between agents.

##### Steps to run
### Prerequisites
1. Windows 10/11
2. Python 3.8
3. GPU that supports Cuda
4. Epic Games Launcher
5. Rocket League (from epic games store)
6. BakkesMod

Note: Check prerequisites for each of the items above

### Setup
We recommend using conda to create your virtual environment
1. Create a virtual environment with **python 3.8** and **pip** - **[If using conda]** run ```conda create -n rocket python=3.8```
2. Activate your environment with ```conda activate __NAME__```
3. Go to ```rocket-learn-master``` folder and run ```pip install .```
4. Go back to the base folder of the project and install the requirements in ```requirements.txt``` with ```pip install -r requirements.txt```
5. Install pywin32 with ```conda install pywin32==228``` - Note: pip usually returns an error while installing, so we use conda

Tip: miniconda3 was used in the project

### Run
Make sure you have **Epic Games** and **BakkesMod** open and then choose one of the following:
1. In the terminal navigate to the project default directory and run ```python run.py```
2. **[If using conda]** Insert your conda virtual environment name in ```runGame.bat```, and double-click the file with the mouse to run

You can change the game settings and used agents/bots in ```rlbot.cfg``` file

Note: The agent will perform poorly if you cannot manage to run the game with at least 120FPS
### Project Structure
- rlbot_agents
  - Agents definitions for evaluations
    - Latest -> 900 million steps of training
    - 800 -> 800 million steps of training
    - 450 -> 450 million steps of training (half of total training of the latest version)
    - 280 -> 280 million steps of training
    - First -> 10 million steps of training
    - Psyonix -> Rocket League best in-game hardcoded bot
- training
  - Agent definition for training
    - aasma_body
      - Components of the agent, including rewards, actions and observations
  - Separation of **worker**, which run the Rocket League game with the current PPO state and produce rollouts to a redis server, and **learner** which collects the provided rollouts from the worker and performs gradient updates
