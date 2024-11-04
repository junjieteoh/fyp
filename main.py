import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import plotly.graph_objects as go
# =========================
# 1. Configuration and Parameters
# =========================

class Config:
    def __init__(self):
        # General settings
        self.num_clients = 200  # Number of clients in the FL system
        self.max_iterations = 1500  # Maximum number of iterations for the simulation
        self.convergence_threshold = 1e-6  # Threshold to determine convergence
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.plot_dir = 'plots_new'
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Strategies: S_H - Honest, S_W - Withholding, S_A - Adversarial
        self.strategies = ['S_H', 'S_W', 'S_A']
        
        # Initial Strategy Distributions
        self.initial_distribution = [0.33, 0.34, 0.33]  # Default even distribution
        
        # Reward Models
        self.reward_models = ['fixed', 'performance', 'fairness_based', 'adaptive']
        self.selected_reward_model = 'fixed'  # Default reward model
        
        # Cost Models
        self.cost_models = ['no_cost', 'computational', 'privacy']
        self.selected_cost_model = 'computational'  # Default cost model
        
        # Cost Structures for Different Models
        self.cost_structures = {
            'no_cost': {'S_H': 0, 'S_W': 0, 'S_A': 0},
            'computational': {'S_H': 10, 'S_W': 5, 'S_A': 2},
            'privacy': {'S_H': 20, 'S_W': 5, 'S_A': 2}
        }
        
        # Additional Factors
        self.latency = 0.0  # Default latency
        self.external_incentive = {'S_H': 0.0, 'S_W': 0.0, 'S_A': 0.0}
        
        # Machine Learning Parameters
        self.base_model_accuracy = 0.5
        self.contribution_factors = {'S_H': 0.6, 'S_W': 0.2, 'S_A': -0.4}
        
        # Adaptive Reward Parameters
        self.eta = 0.05  # Learning rate for adaptive rewards
        
        # Penalty scaling factor
        self.alpha = 0.0
        
        # Convergence Analysis Parameters
        self.simplex_points = 10  # Number of points per dimension for simplex sampling
        
        # Seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)

config = Config()

# =========================
# 2. Helper Classes and Functions
# =========================

class Client:
    def __init__(self, client_id, strategy, config, cost_structure):
        self.client_id = client_id
        self.strategy = strategy
        self.config = config
        self.cost_structure = cost_structure
        self.contribution = 0.0
        self.cost = 0.0
        self.reward = 0.0
        self.payoff = 0.0
        
    def compute_contribution(self):
        self.contribution = self.config.contribution_factors[self.strategy]
        
    def compute_cost(self):
        base_cost = self.cost_structure[self.strategy]
        latency_cost = self.config.latency * base_cost
        total_cost = base_cost + latency_cost
        external_incentive = self.config.external_incentive.get(self.strategy, 0.0)
        self.cost = total_cost - external_incentive
        return self.cost
        
    def compute_payoff(self):
        self.payoff = self.reward - self.cost

def generate_simplex_points(n_points=20):
    """
    Generate a fine grid of points in the 2-simplex (triangle) where coordinates sum to 1.
    """
    points = []
    step = 1.0 / (n_points - 1)
    
    for i in range(n_points):
        for j in range(n_points - i):
            x = i * step
            y = j * step
            z = 1 - x - y
            if z >= -0.0001:  # Allow for small numerical errors
                points.append([x, y, z])
    
    return np.array(points)

def compute_rewards(clients, config, R):
    if config.selected_reward_model == 'fixed':
        total_reward = R
        for client in clients:
            client.reward = total_reward / len(clients)
    elif config.selected_reward_model == 'performance':
        total_reward = R
        total_positive_contribution = sum(max(0, client.contribution) for client in clients)
        for client in clients:
            positive_contribution = max(0, client.contribution)
            client.reward = total_reward * (positive_contribution / (total_positive_contribution + config.epsilon))
    elif config.selected_reward_model == 'fairness_based':
        total_reward = R
        contributions = [max(0, client.contribution) for client in clients]
        shapley_values = compute_shapley_values(contributions)
        for client, shapley_value in zip(clients, shapley_values):
            client.reward = total_reward * shapley_value
    elif config.selected_reward_model == 'adaptive':
        pass
    return

def compute_shapley_values(contributions):
    total_contribution = sum(contributions)
    shapley_values = []
    for contribution in contributions:
        shapley_value = contribution / (total_contribution + 1e-8)
        shapley_values.append(shapley_value)
    return shapley_values

def update_strategy_proportions(clients, config):
    payoffs = np.array([client.payoff for client in clients])
    avg_payoff = np.mean(payoffs)
    
    if config.alpha > 0:
        for client in clients:
            if client.payoff < avg_payoff:
                penalty = config.alpha * ((avg_payoff - client.payoff) / (avg_payoff + config.epsilon)) ** 2
                client.payoff -= penalty
        payoffs = np.array([client.payoff for client in clients])
        avg_payoff = np.mean(payoffs)
    
    min_payoff = np.min(payoffs)
    if min_payoff < 0:
        payoffs = payoffs - min_payoff + config.epsilon
        avg_payoff = np.mean(payoffs)
    
    x_i = np.zeros(len(config.strategies))
    Pi_i = np.zeros(len(config.strategies))
    
    for idx, strategy in enumerate(config.strategies):
        strategy_indices = [client.strategy == strategy for client in clients]
        x_i[idx] = np.sum(strategy_indices) / config.num_clients
        strategy_payoffs = payoffs[strategy_indices]
        if len(strategy_payoffs) == 0:
            Pi_i[idx] = 0.0
        else:
            Pi_i[idx] = np.mean(strategy_payoffs)
    
    numerator = x_i * (Pi_i - avg_payoff)
    new_proportions = x_i + numerator
    new_proportions = np.clip(new_proportions, 0.0, 1.0)
    total = np.sum(new_proportions)
    if total > 0:
        new_proportions = new_proportions / total
    else:
        num_strategies = len(config.strategies)
        new_proportions = np.ones(num_strategies) / num_strategies
    
    return new_proportions.tolist()

def simulate_global_model_performance(clients, config):
    total_contribution = sum(client.contribution for client in clients)
    max_possible_contribution = config.num_clients * max(config.contribution_factors.values())
    performance = config.base_model_accuracy + (total_contribution / (max_possible_contribution + config.epsilon)) * (1 - config.base_model_accuracy)
    performance += np.random.normal(0, 0.01)
    performance = np.clip(performance, 0.0, 1.0)
    return performance

def update_clients_strategies(clients, new_strategy_proportions, config):
    new_strategies = []
    for proportion, strategy in zip(new_strategy_proportions, config.strategies):
        count = int(round(proportion * config.num_clients))
        new_strategies.extend([strategy] * count)
    while len(new_strategies) < config.num_clients:
        new_strategies.append(config.strategies[0])
    while len(new_strategies) > config.num_clients:
        new_strategies.pop()
    np.random.shuffle(new_strategies)
    for client, new_strategy in zip(clients, new_strategies):
        client.strategy = new_strategy

def analyze_convergence(config, experiment_params, n_points=10):
    """
    Analyze convergence from different starting points in the strategy space.
    """
    starting_points = generate_simplex_points(n_points)
    results_data = []
    
    for start_point in starting_points:
        experiment_params['initial_distribution'] = start_point.tolist()
        results = run_simulation(config, experiment_params)
        
        results_data.append({
            'start_S_H': start_point[0],
            'start_S_W': start_point[1],
            'start_S_A': start_point[2],
            'end_S_H': results['final_proportions'][0],
            'end_S_W': results['final_proportions'][1],
            'end_S_A': results['final_proportions'][2],
            'iterations': results['iterations'],
            'final_performance': results['performance_history'][-1]
        })
    
    return pd.DataFrame(results_data)

def plot_convergence_analysis(df, experiment_name, config):
    """
    Create visualizations of the convergence analysis.
    """
    plot_dir = os.path.join(config.plot_dir, 'convergence_analysis', experiment_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df['start_S_H'], df['start_S_W'], df['start_S_A'], 
              c='blue', marker='o', label='Start', alpha=0.6)
    ax.scatter(df['end_S_H'], df['end_S_W'], df['end_S_A'], 
              c='red', marker='^', label='End', alpha=0.6)
    
    for _, row in df.iterrows():
        ax.plot([row['start_S_H'], row['end_S_H']],
                [row['start_S_W'], row['end_S_W']],
                [row['start_S_A'], row['end_S_A']],
                'gray', alpha=0.3)
    
    ax.set_xlabel('S_H')
    ax.set_ylabel('S_W')
    ax.set_zlabel('S_A')
    ax.set_title(f'Strategy Space Convergence\n{experiment_name}')
    ax.legend()
    
    plt.savefig(os.path.join(plot_dir, 'convergence_3d.png'))
    plt.close()
    
    # Heat map
    plt.figure(figsize=(10, 8))
    plt.scatter(df['start_S_H'], df['start_S_W'], 
                c=df['iterations'], cmap='viridis')
    plt.colorbar(label='Iterations to Convergence')
    plt.xlabel('Initial S_H Proportion')
    plt.ylabel('Initial S_W Proportion')
    plt.title(f'Iterations to Convergence\n{experiment_name}')
    plt.savefig(os.path.join(plot_dir, 'convergence_heatmap.png'))
    plt.close()
    
    # Save data
    df.to_csv(os.path.join(plot_dir, 'convergence_data.csv'), index=False)

def run_simulation(config, experiment_params):
    """
    Run a simulation with the given configuration and parameters.
    """
    initial_distribution = experiment_params.get('initial_distribution', config.initial_distribution)
    cost_structure_name = experiment_params.get('cost_structure_name', config.selected_cost_model)
    latency = experiment_params.get('latency', config.latency)
    external_incentive = experiment_params.get('external_incentive', config.external_incentive)
    eta = experiment_params.get('eta', config.eta)
    experiment_name = experiment_params.get('experiment_name', 'default_experiment')
    
    cost_structure = config.cost_structures[cost_structure_name]
    config.latency = latency
    config.external_incentive = external_incentive
    config.eta = eta
    
    clients = []
    initial_counts = [int(round(p * config.num_clients)) for p in initial_distribution]
    while sum(initial_counts) < config.num_clients:
        initial_counts[0] += 1
    while sum(initial_counts) > config.num_clients:
        initial_counts[0] -= 1
    
    client_id = 0
    for strategy, count in zip(config.strategies, initial_counts):
        for _ in range(count):
            clients.append(Client(client_id, strategy, config, cost_structure))
            client_id += 1
    
    np.random.shuffle(clients)
    
    strategy_proportions_history = []
    performance_history = []
    reward_pool_history = []
    
    R = 1000  # Initial reward pool
    
    initial_proportions = [sum(client.strategy == s for client in clients)/config.num_clients 
                          for s in config.strategies]
    strategy_proportions_history.append(initial_proportions.copy())
    
    iteration = 0
    converged = False
    
    while not converged and iteration < config.max_iterations:
        for client in clients:
            client.compute_contribution()
            client.compute_cost()
        
        compute_rewards(clients, config, R)
        
        for client in clients:
            client.compute_payoff()
        
        new_strategy_proportions = update_strategy_proportions(clients, config)
        strategy_proportions_history.append(new_strategy_proportions.copy())
        
        performance = simulate_global_model_performance(clients, config)
        performance_history.append(performance)
        
        if config.selected_reward_model == 'adaptive':
            delta_R = config.eta * (performance - config.base_model_accuracy)
            R = R * (1 + delta_R)
            R = max(R, 0.0)
        
        reward_pool_history.append(R)
        
        if iteration > 0:
            prev_proportions = np.array(strategy_proportions_history[-2])
            curr_proportions = np.array(strategy_proportions_history[-1])
            diff = np.sum(np.abs(curr_proportions - prev_proportions))
            if diff < config.convergence_threshold:
                converged = True
        
        update_clients_strategies(clients, new_strategy_proportions, config)
        
        iteration += 1
    
    results = {
        'strategy_history': np.array(strategy_proportions_history),
        'performance_history': performance_history,
        'reward_pool_history': reward_pool_history,
        'iterations': iteration,
        'experiment_name': experiment_name,
        'final_proportions': new_strategy_proportions
    }
    return results

def run_convergence_analysis(config):
    """
    Run simulations from many starting points and create visualizations.
    """
    # Generate experiment name based on current configuration
    experiment_name = f"RQ{config.current_rq}_{config.selected_reward_model}_{config.selected_cost_model}"
    if config.selected_reward_model == 'adaptive':
        experiment_name += f"_eta_{config.eta}"
    if config.alpha > 0:
        experiment_name += f"_alpha_{config.alpha}"
    if config.latency > 0:
        experiment_name += f"_latency_{config.latency}"
    if any(v != 0 for v in config.external_incentive.values()):
        experiment_name += "_with_incentives"
    
    print(f"Running convergence analysis for: {experiment_name}")
    
    # Generate fine grid of starting points
    starting_points = generate_simplex_points(30)
    
    # Store results
    all_trajectories = []
    final_points = []
    
    # Run simulation from each starting point
    for start_point in starting_points:
        print(f"Simulating from starting point: {start_point}")
        
        experiment_params = {
            'initial_distribution': start_point.tolist(),
            'cost_structure_name': config.selected_cost_model,
            'latency': config.latency,
            'external_incentive': config.external_incentive,
            'experiment_name': experiment_name
        }
        
        results = run_simulation(config, experiment_params)
        trajectory = results['strategy_history']
        all_trajectories.append(trajectory)
        final_points.append(trajectory[-1])

    # Create plot directory with experiment-specific subfolder
    plot_dir = os.path.join(config.plot_dir, 'convergence_analysis', 
                           f'RQ{config.current_rq}_{config.experiment_subfolder}',
                           experiment_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create high-res static 3D plot
    plt.figure(figsize=(24, 24), dpi=300)
    ax = plt.axes(projection='3d')
    
    # Plot trajectories, starting points, and endpoints
    for trajectory in all_trajectories:
        # Plot tiny light green starting point
        start_point = trajectory[0]
        ax.scatter(start_point[0], start_point[1], start_point[2],
                  color='green', s=10, alpha=0.3)
        
        # Plot trajectory with gradually increasing darkness
        num_points = len(trajectory)
        for i in range(num_points - 1):
            # Calculate darkness level (0.1 to 0.9) based on position in trajectory
            alpha = 0.4 + (0.6 * (i / (num_points-1)))
            
            # Plot segment with gradient color
            segment = trajectory[i:i+2]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                   color='blue', linewidth=0.5, alpha=alpha)
        
        # Plot the final endpoint as a red dot
        end_point = trajectory[-1]
        ax.scatter(end_point[0], end_point[1], end_point[2],
                  color='red', s=50)
    
    # Add simplex triangle outline
    triangle_points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    ax.plot(triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2],
            'k-', linewidth=1, alpha=0.2, label='Strategy Space')
    # Set optimal viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Customize plot
    ax.set_xlabel('S_H Proportion', fontsize=16)
    ax.set_ylabel('S_W Proportion', fontsize=16)
    ax.set_zlabel('S_A Proportion', fontsize=16)
    
    # Create detailed title with configuration details
    title = f'Strategy Evolution - RQ{config.current_rq}\n'
    title += f'Reward Model: {config.selected_reward_model} (Initial Pool: 1000)\n'
    title += f'Cost Model: {config.selected_cost_model}\n'
    title += f'Cost Structure: {config.cost_structures[config.selected_cost_model]}\n'
    title += f'Contribution Factors: {config.contribution_factors}\n'
    if config.selected_reward_model == 'adaptive':
        title += f'η: {config.eta}\n'
    if config.alpha > 0:
        title += f'α: {config.alpha}\n'
    if config.latency > 0:
        title += f'Latency: {config.latency}\n'
    if any(v != 0 for v in config.external_incentive.values()):
        title += f'External Incentives: {config.external_incentive}\n'
    
    ax.set_title(title, fontsize=20, pad=20)
    
    # Add legend
    ax.scatter([], [], [], color='green', s=2, alpha=0.8, label='Starting Points')
    ax.scatter([], [], [], color='red', s=50, label='Convergence Points')
    ax.plot([], [], [], color='blue', linewidth=0.2, alpha=0.3, label='Trajectories')
    ax.plot([], [], [], color='purple', linewidth=0.2, alpha=0.6, label='Final Approach')
    ax.legend(fontsize=14)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    
    # Save static plot
    plt.savefig(os.path.join(plot_dir, f'RQ{config.current_rq}_{config.selected_reward_model}.png'), 
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()

    # Create interactive plotly version
    fig = go.Figure()
    
    # Add trajectories
    for trajectory in all_trajectories:
        # Add starting point
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0][0]], 
            y=[trajectory[0][1]], 
            z=[trajectory[0][2]],
            mode='markers',
            marker=dict(size=3, color='green'),
            name='Starting Point',
            showlegend=False
        ))
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1], 
            z=trajectory[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Trajectory',
            showlegend=False
        ))
        
        # Add endpoint
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1][0]],
            y=[trajectory[-1][1]],
            z=[trajectory[-1][2]],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Convergence Point',
            showlegend=False
        ))
    
    # Add simplex triangle
    fig.add_trace(go.Scatter3d(
        x=triangle_points[:, 0],
        y=triangle_points[:, 1],
        z=triangle_points[:, 2],
        mode='lines',
        line=dict(color='black', width=3),
        name='Strategy Space'
    ))
    
    # Create detailed title for interactive plot
    interactive_title = f'Strategy Evolution - RQ{config.current_rq}<br>'
    interactive_title += f'Reward Model: {config.selected_reward_model} (Initial Pool: 1000)<br>'
    interactive_title += f'Cost Model: {config.selected_cost_model}<br>'
    interactive_title += f'Cost Structure: {config.cost_structures[config.selected_cost_model]}<br>'
    interactive_title += f'Contribution Factors: {config.contribution_factors}<br>'
    if config.selected_reward_model == 'adaptive':
        interactive_title += f'η: {config.eta}<br>'
    if config.alpha > 0:
        interactive_title += f'α: {config.alpha}<br>'
    if config.latency > 0:
        interactive_title += f'Latency: {config.latency}<br>'
    if any(v != 0 for v in config.external_incentive.values()):
        interactive_title += f'External Incentives: {config.external_incentive}<br>'
    
    # Update layout
    fig.update_layout(
        title=interactive_title,
        scene=dict(
            xaxis_title='S_H Proportion',
            yaxis_title='S_W Proportion',
            zaxis_title='S_A Proportion',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True
    )
    
    # Save interactive plot
    fig.write_html(os.path.join(plot_dir, f'RQ{config.current_rq}_{config.selected_reward_model}_interactive.html'))

    # Save data
    df = pd.DataFrame({
        'start_S_H': [t[0][0] for t in all_trajectories],
        'start_S_W': [t[0][1] for t in all_trajectories],
        'start_S_A': [t[0][2] for t in all_trajectories],
        'end_S_H': [t[-1][0] for t in all_trajectories],
        'end_S_W': [t[-1][1] for t in all_trajectories],
        'end_S_A': [t[-1][2] for t in all_trajectories]
    })
    df.to_csv(os.path.join(plot_dir, 'convergence_data.csv'), index=False)

    # Print summary
    final_points = np.array(final_points)
    print("\nConvergence Analysis Summary:")
    print(f"Number of starting points: {len(starting_points)}")
    print(f"Number of unique convergence points: {len(np.unique(final_points, axis=0))}")

def run_experiments(config):
    """
    Run comprehensive convergence analysis for different experimental conditions.
    """
    # Base experiment parameters
    base_params = {
        'cost_structure_name': 'computational',
        'latency': 0.0,
        'external_incentive': {'S_H': 0.0, 'S_W': 0.0, 'S_A': 0.0},
        'eta': 0.05
    }

    # RQ1: Reward Schemes Influence
    config.current_rq = 1
    config.experiment_subfolder = "reward_schemes"
    for reward_model in ['fixed', 'performance', 'fairness_based', 'adaptive']:
        config.selected_reward_model = reward_model
        if reward_model == 'adaptive':
            for eta in [0.01, 0.05, 0.1]:
                config.eta = eta
                experiment_name = f'convergence_RQ1_{reward_model}_eta_{eta}'
                print(f"Analyzing convergence for {experiment_name}")
                run_convergence_analysis(config)
        else:
            experiment_name = f'convergence_RQ1_{reward_model}'
            print(f"Analyzing convergence for {experiment_name}")
            run_convergence_analysis(config)

    # RQ2: Cost Structures Impact
    config.current_rq = 2
    config.experiment_subfolder = "cost_structures"
    config.selected_reward_model = 'performance'  # Keep reward model constant
    for cost_model in config.cost_models:
        config.selected_cost_model = cost_model
        experiment_name = f'convergence_RQ2_cost_{cost_model}'
        print(f"Analyzing convergence for {experiment_name}")
        run_convergence_analysis(config)

    # Cost scaling variations
    config.experiment_subfolder = "cost_scaling"
    cost_variations = [
        {'S_H': 15, 'S_W': 5, 'S_A': 2},
        {'S_H': 20, 'S_W': 5, 'S_A': 2},
        {'S_H': 25, 'S_W': 5, 'S_A': 2}
    ]
    for idx, cost_structure in enumerate(cost_variations):
        cost_name = f'custom_cost_{idx}'
        config.cost_structures[cost_name] = cost_structure
        config.selected_cost_model = cost_name
        experiment_name = f'convergence_RQ2_scaling_{cost_structure["S_H"]}'
        print(f"Analyzing convergence for {experiment_name}")
        run_convergence_analysis(config)

    # RQ3: Additional Factors
    config.current_rq = 3
    config.selected_reward_model = 'performance'
    config.selected_cost_model = 'computational'
    
    # Latency impact
    config.experiment_subfolder = "latency_impact"
    for latency in [0.0, 0.1, 0.2]:
        config.latency = latency
        experiment_name = f'convergence_RQ3_latency_{latency}'
        print(f"Analyzing convergence for {experiment_name}")
        run_convergence_analysis(config)

    # External incentives
    config.experiment_subfolder = "external_incentives"
    incentive_variations = [
        {'S_H': 0.0, 'S_W': 0.0, 'S_A': -10.0},
        {'S_H': 5.0, 'S_W': 0.0, 'S_A': -5.0}
    ]
    for idx, incentive in enumerate(incentive_variations):
        config.external_incentive = incentive
        experiment_name = f'convergence_RQ3_incentive_{idx}'
        print(f"Analyzing convergence for {experiment_name}")
        run_convergence_analysis(config)

    # RQ4: Fairness Integration
    config.current_rq = 4
    config.experiment_subfolder = "fairness_integration"
    # Long-term analysis with fairness-based rewards
    config.selected_reward_model = 'fairness_based'
    config.max_iterations = 1000
    experiment_name = 'convergence_RQ4_longterm_fairness'
    print(f"Analyzing convergence for {experiment_name}")
    run_convergence_analysis(config)
    config.max_iterations = 500  # Reset

    # RQ5: Multiple Equilibria with Penalties
    config.current_rq = 5
    config.experiment_subfolder = "equilibria_penalties"
    config.selected_reward_model = 'adaptive'
    for alpha in [0.0, 0.3, 0.5]:
        config.alpha = alpha
        for eta in [0.01, 0.05, 0.1]:
            config.eta = eta
            experiment_name = f'convergence_RQ5_alpha_{alpha}_eta_{eta}'
            print(f"Analyzing convergence for {experiment_name}")
            run_convergence_analysis(config)
    
    # Reset config to default values
    config.alpha = 0.0
    config.eta = 0.05
    config.latency = 0.0
    config.external_incentive = {'S_H': 0.0, 'S_W': 0.0, 'S_A': 0.0}
    config.selected_reward_model = 'fixed'
    config.selected_cost_model = 'computational'

if __name__ == "__main__":
    config = Config()
    run_experiments(config)