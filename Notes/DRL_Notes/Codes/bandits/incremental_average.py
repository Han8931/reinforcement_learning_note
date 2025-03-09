def incremental_average(rewards):
    """
    Given a list of rewards, compute the incremental average Q_n
    after each new reward using:
    
        Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
    
    Args:
        rewards (list of float): A list of reward values R_i.

    Returns:
        list of float: The running averages Q_1, Q_2, ..., Q_n.
    """
    Q_values = []
    Q_current = 0.0  # Initial estimate (can be set to zero or any prior)
    
    for i, R in enumerate(rewards, start=1):
        # i is the current step (n), R is R_n
        # Update Q_{n+1} using the incremental formula
        Q_current = Q_current + (1.0 / i) * (R - Q_current)
        
        # Store the updated estimate
        Q_values.append(Q_current)
    
    return Q_values

# Example usage:
rewards_example = [10, 20, 15, 5, 30]
running_estimates = incremental_average(rewards_example)

for step, (reward, Q) in enumerate(zip(rewards_example, running_estimates), start=1):
    print(f"Step {step}, Reward = {reward}, Running Average = {Q:.2f}")
