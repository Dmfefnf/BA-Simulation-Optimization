# Table of Contents

## Introduction
- State of Research
- Research Question
- Hypothesis

## Theoretical Background

### Discrete Event Simulation
- Discrete Event Simulation in Manufacturing Systems
- Simulation of dispatch Scheduling

### Simulation Optimization
- Simulation Optimization Techniques
- Simulation Optimization Challenges
- Benefits of Simulation Optimization

### Optimization Algorithms

#### Bayesian Optimization

#### Reinforcement Learning
- Reinforcement Learning in Manufacturing Systems
- What possible RL algorithms can be used for simulation optimization?
- Q-learning Algorithm

#### Interplay BO and RL
- If theory available, discuss the interplay between BO and RL in simulation optimization
-> If not, first mention in the introduction that this is a research gap and then discuss it in the methodology section, where you explain how you will use BO to tune the hyperparameters of RL. 

## Methodology

### Simulation Model
- Base Simulation Model
- Simulation Model Parameters
- BO Simulation Model
- RL Simulation Model

### Hypothesis and Experimental Design

### Optimization Algorithms

#### Bayesian Optimization 
- Implementation
    - Ax Platform
- Experimental Setup
    - Single Run
    - Multiple Runs and Statistical Analysis
    - BO vs Random Search

#### Reinforcement Learning Implementation
- Implementation
    - Action Space Design
        - FIFO
        - EDD
        - Longest Waiting Time
        - Highest Lateness Risk
    - Q-learning Algorithm
        - State Space
        - Action Space
        - Reward Function 
    - Hyperparameter Tuning
- Experimental Setup 
    - Baseline Comparison
    - Random Agent Comparison
    - Training and Evaluation Protocol

## Results

### Bayesian Optimization Results

### Reinforcement Learning Results
- BO in RL Hyperparameter Tuning
- Pure RL Performance

## Discussion
- Interpretation of Results
    - Bayesian Optimization Results
    - Reinforcement Learning Results
- Meaning of Results for Optimization Methods
- Implications for Industrial Practice
- Limitations and Future Work

## Conclusion
