import pickle
import numpy as np
import matplotlib.pyplot as plt


def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    try:
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        # Corrected keys based on file content
        Y_train = data['train_set']
        Y_test = data['test_set']

        with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
            scenario_data = pickle.load(f)
        # Corrected keys based on file content
        Prophets_scenario = scenario_data['train_set']
        True_Risk = np.array(scenario_data['true_risk'])  # [0.2, 0.4]
    except FileNotFoundError:
        print("Error: Could not find one or more of the required pickle files.")
        return 0, 0, 0, 0

    # --- Setup ---
    NUM_EXPERIMENTS = 100
    NUM_PROPHETS = 2

    # The 'best' prophet has the lowest True Risk (0.2)
    BEST_PROPHET_ID = np.argmin(True_Risk)  # Prophet 1 (ID 0)

    # Approximation Error is the True Risk of the best available prophet
    APPROXIMATION_ERROR = True_Risk[BEST_PROPHET_ID]

    # --- Storage ---
    selected_prophet_true_errors = []
    best_prophet_chosen_count = 0

    # --- Run Experiments ---
    for _ in range(NUM_EXPERIMENTS):
        # 1. Randomly select a single game for the training set (N=1)
        train_game_id = np.random.randint(0, len(Y_train))
        y_train_game = Y_train[train_game_id]

        # Calculate Empirical Risk (ER) for each prophet on the single game
        prophet_predictions = Prophets_scenario[:, train_game_id]

        # ER = 1 if prediction != true_outcome, 0 otherwise
        empirical_risks = (prophet_predictions.astype(int) != y_train_game.astype(int)).astype(int)

        # Select prophet based on ERM (minimize Empirical Risk)
        min_er = np.min(empirical_risks)
        best_prophet_candidates = np.where(empirical_risks == min_er)[0]

        # In case of a tie, sample uniformly from the tied prophets
        selected_prophet_id = np.random.choice(best_prophet_candidates)

        # Get the True Error (True Risk) of the selected prophet
        selected_prophet_true_error = True_Risk[selected_prophet_id]
        selected_prophet_true_errors.append(selected_prophet_true_error)

        # Check if the BEST prophet (ID 0) was chosen
        if selected_prophet_id == BEST_PROPHET_ID:
            best_prophet_chosen_count += 1

    # --- Final Calculations ---
    average_error_of_selected_prophets = np.mean(selected_prophet_true_errors)

    # Mean Estimation Error
    mean_estimation_error = average_error_of_selected_prophets - APPROXIMATION_ERROR

    # --- Print Results (for the report) ---
    print(f"Average Error of selected prophets (over 100 experiments): {average_error_of_selected_prophets:.4f}")
    print(f"Number of times the best prophet was chosen: {best_prophet_chosen_count} out of {NUM_EXPERIMENTS}")
    print(f"Approximation Error (True Risk of best prophet): {APPROXIMATION_ERROR:.4f}")
    print(f"Mean Estimation Error: {mean_estimation_error:.4f}")

    return average_error_of_selected_prophets, best_prophet_chosen_count, APPROXIMATION_ERROR, mean_estimation_error

def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    # --- Load Data (Same as Scenario 1) ---
    try:
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        Y_train = data['train_set']
        Y_test = data['test_set']

        with open('scenario_one_and_two_prophets.pkl', 'rb') as f:
            scenario_data = pickle.load(f)
        Prophets_scenario = scenario_data['train_set']
        True_Risk = np.array(scenario_data['true_risk'])  # [0.2, 0.4]
    except FileNotFoundError:
        print("Error: Could not find one or more of the required pickle files.")
        return 0, 0, 0, 0

    # --- Setup ---
    NUM_EXPERIMENTS = 100
    N_GAMES = 10  # <--- The key change for this scenario

    # The 'best' prophet (P1) has the lowest True Risk (0.2)
    BEST_PROPHET_ID = np.argmin(True_Risk)
    APPROXIMATION_ERROR = True_Risk[BEST_PROPHET_ID]

    # --- Storage ---
    selected_prophet_true_errors = []
    best_prophet_chosen_count = 0

    # --- Run Experiments (100 times) ---
    for _ in range(NUM_EXPERIMENTS):
        # 1. Randomly sample 10 game indices for the training set (N=10)
        train_game_ids = np.random.choice(len(Y_train), size=N_GAMES, replace=False)
        y_train_games = Y_train[train_game_ids]

        # 2. Get the predictions for those 10 games from both prophets
        # Shape: [2 prophets, 10 games]
        prophet_predictions = Prophets_scenario[:, train_game_ids]

        # 3. Calculate Empirical Risk (ER) for both prophets on the N=10 sample

        # Calculate losses (0/1 for each game, shape [2, 10])
        losses = (prophet_predictions.astype(int) != y_train_games.astype(int))

        # ER is the average loss over the 10 games (shape [2])
        empirical_risks = np.mean(losses, axis=1)  # Average error across the 10 columns

        # 4. Apply ERM: Select prophet based on minimum Empirical Risk
        min_er = np.min(empirical_risks)
        best_prophet_candidates = np.where(empirical_risks == min_er)[0]

        # Break ties uniformly
        selected_prophet_id = np.random.choice(best_prophet_candidates)

        # 5. Record the True Risk of the selected prophet
        selected_prophet_true_error = True_Risk[selected_prophet_id]
        selected_prophet_true_errors.append(selected_prophet_true_error)

        if selected_prophet_id == BEST_PROPHET_ID:
            best_prophet_chosen_count += 1

    # --- Final Calculations ---
    average_error_of_selected_prophets = np.mean(selected_prophet_true_errors)
    mean_estimation_error = average_error_of_selected_prophets - APPROXIMATION_ERROR

    # --- Print Results (for the report) ---

    print(f"Average Error of selected prophets (N={N_GAMES} experiments): {average_error_of_selected_prophets:.4f}")
    print(f"Number of times the best prophet was chosen: {best_prophet_chosen_count} out of {NUM_EXPERIMENTS}")
    print(f"Approximation Error: {APPROXIMATION_ERROR:.4f}")
    print(f"Mean Estimation Error: {mean_estimation_error:.4f}")

    return average_error_of_selected_prophets, best_prophet_chosen_count, APPROXIMATION_ERROR, mean_estimation_error



# --- HELPER 1: Data Loading Function ---
def load_data_and_prophets(prophet_file_name):
    """
    Loads general game data and scenario-specific prophet data using correct keys.
    Returns: Y_train, True_Risk, Prophets_scenario (predictions)
    (Y_train is the true outcome array for training games)
    """
    try:
        # Load general game data
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        Y_train = data['train_set']  # Corrected key for true outcomes

        # Load scenario-specific prophet data (e.g., Many Prophets)
        with open(prophet_file_name, 'rb') as f:
            scenario_data = pickle.load(f)

        Prophets_scenario = scenario_data['train_set']  # Corrected key for predictions
        True_Risk = np.array(scenario_data['true_risk'])  # Corrected key for risks

        return Y_train, True_Risk, Prophets_scenario
    except FileNotFoundError:
        print(f"Error: Could not find required file: {prophet_file_name}.")
        return None, None, None
    except KeyError as e:
        print(f"Error: Missing expected key: {e}. Check the file structure.")
        return None, None, None


# --- CORE LOGIC FUNCTION ---
def run_erm_experiment(Y_train, True_Risk, Prophets_scenario, N_GAMES, NUM_EXPERIMENTS=100,
                       calculate_one_percent_metric=False):
    """
    Runs the core ERM simulation logic for any number of prophets and any sample size N.
    """
    if Y_train is None:
        return 0, 0, 0, 0, 0  # Return zeros on failure

    # --- Setup ---
    BEST_PROPHET_ID = np.argmin(True_Risk)
    APPROXIMATION_ERROR = True_Risk[BEST_PROPHET_ID]

    # --- Storage ---
    selected_prophet_true_errors = []
    best_prophet_chosen_count = 0
    not_1_percent_worse_count = 0

    # --- Run Experiments ---
    for _ in range(NUM_EXPERIMENTS):
        # 1. Sample N games (handles N=10 and N=1000)
        # replace=False ensures unique indices
        train_game_ids = np.random.choice(len(Y_train), size=N_GAMES, replace=False)
        y_train_games = Y_train[train_game_ids]

        # 2. Get predictions (Shape: [N_prophets, N_GAMES])
        prophet_predictions = Prophets_scenario[:, train_game_ids]

        # 3. Calculate Empirical Risk (ER)
        # a) Calculate 0/1 losses for all prophets on all N games
        losses = (prophet_predictions.astype(int) != y_train_games.astype(int))

        # b) Calculate mean loss across the N games (axis=1) -> Empirical Risk
        empirical_risks = np.mean(losses, axis=1)

        # 4. Apply ERM: Select prophet based on minimum Empirical Risk
        min_er = np.min(empirical_risks)

        # Use tolerance (1e-9) for robust floating-point comparison
        tolerance = 1e-9
        best_prophet_candidates = np.where(empirical_risks <= min_er + tolerance)[0]

        # Break ties uniformly
        selected_prophet_id = np.random.choice(best_prophet_candidates)

        # 5. Record the True Risk
        selected_prophet_true_error = True_Risk[selected_prophet_id]
        selected_prophet_true_errors.append(selected_prophet_true_error)

        if selected_prophet_id == BEST_PROPHET_ID:
            best_prophet_chosen_count += 1

        # 6. Calculate 1% metric (Required for Scenarios 3 and 4)
        if calculate_one_percent_metric and selected_prophet_true_error <= APPROXIMATION_ERROR + 0.01:
            not_1_percent_worse_count += 1

    # --- Final Calculations ---
    average_error_of_selected_prophets = np.mean(selected_prophet_true_errors)
    mean_estimation_error = average_error_of_selected_prophets - APPROXIMATION_ERROR

    # --- Print Results ---
    print(f"Approximation Error: {APPROXIMATION_ERROR:.4f}")
    print(f"Average Error of selected prophets: {average_error_of_selected_prophets:.4f}")
    print(f"Mean Estimation Error: {mean_estimation_error:.4f}")
    print(f"Number of times the BEST prophet was chosen: {best_prophet_chosen_count} out of {NUM_EXPERIMENTS}")
    print(f"Times NOT 1% worse than the best: {not_1_percent_worse_count} out of {NUM_EXPERIMENTS}")

    return average_error_of_selected_prophets, best_prophet_chosen_count, not_1_percent_worse_count, APPROXIMATION_ERROR, mean_estimation_error


# --- SCENARIO 3: Many Prophets, Small Data N=10 ---
def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    print("\n--- Scenario 3: Many Prophets, 10 Games ---")
    Y_train, True_Risk, Prophets_scenario = load_data_and_prophets('scenario_three_and_four_prophets.pkl')

    # Call the modular function with N_GAMES=10 and the flag for 1% metric
    run_erm_experiment(Y_train, True_Risk, Prophets_scenario, N_GAMES=10, calculate_one_percent_metric=True)


# --- SCENARIO 4: Many Prophets, Many Games N=1000 ---
def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    print("\n--- Scenario 4: Many Prophets, 1000 Games ---")
    Y_train, True_Risk, Prophets_scenario = load_data_and_prophets('scenario_three_and_four_prophets.pkl')

    # Call the modular function with the required N_GAMES=1000
    run_erm_experiment(Y_train, True_Risk, Prophets_scenario, N_GAMES=1000, calculate_one_percent_metric=True)


# --- HELPER 1: Data Loading ---
def load_scenario_5_data(prophet_file_name):
    """
    Loads general game data and the master pool of high-quality prophets [0, 0.2].
    """
    try:
        # Load general game data
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        Y_train = data['train_set']

        # Load scenario-specific prophet data
        with open(prophet_file_name, 'rb') as f:
            scenario_data = pickle.load(f)

        # Keys are assumed to be consistent across all prophet files
        Prophets_master = scenario_data['train_set']
        True_Risk_master = np.array(scenario_data['true_risk'])

        return Y_train, True_Risk_master, Prophets_master
    except Exception as e:
        print(f"Error loading data for Scenario 5: {e}")
        return None, None, None


# --- CORE LOGIC FUNCTION ---
def run_erm_experiment_5(Y_train, True_Risk_subset, Prophets_subset, N_GAMES, NUM_EXPERIMENTS=100):
    """
    Runs the core ERM simulation for a fixed subset of k prophets over m=N_GAMES.
    """

    # 1. Calculate the Approximation Error for this fixed k-subset
    BEST_PROPHET_ID_FULL = np.argmin(True_Risk_subset)
    APPROXIMATION_ERROR_FULL = True_Risk_subset[BEST_PROPHET_ID_FULL]

    # --- Storage ---
    selected_prophet_true_errors = []

    # --- Run Experiments (100 trials for variance reduction) ---
    for _ in range(NUM_EXPERIMENTS):
        # a) Sample N_GAMES (m) games
        train_game_ids = np.random.choice(len(Y_train), size=N_GAMES, replace=False)
        y_train_games = Y_train[train_game_ids]

        # b) Get predictions (Shape: [k, m])
        prophet_predictions = Prophets_subset[:, train_game_ids]

        # c) Calculate Empirical Risk (ER) - Mean loss across the N games
        losses = (prophet_predictions.astype(int) != y_train_games.astype(int))

        # Handle N_GAMES=1 case vs N_GAMES > 1 (using float conversion for safety)
        if N_GAMES > 1:
            empirical_risks = np.mean(losses, axis=1)
            tolerance = 1e-9  # Floating point tolerance
        else:
            empirical_risks = losses.astype(float)
            tolerance = 0

        # d) Apply ERM: Select prophet based on minimum Empirical Risk
        min_er = np.min(empirical_risks)
        best_prophet_candidates = np.where(empirical_risks <= min_er + tolerance)[0]

        # Break ties uniformly
        selected_prophet_id = np.random.choice(best_prophet_candidates)

        # e) Record the True Risk
        selected_prophet_true_error = True_Risk_subset[selected_prophet_id]
        selected_prophet_true_errors.append(selected_prophet_true_error)

    # --- Final Metric Calculations ---
    average_error_of_selected_prophets = np.mean(selected_prophet_true_errors)
    mean_estimation_error = average_error_of_selected_prophets - APPROXIMATION_ERROR_FULL

    return {
        'Avg_Error': average_error_of_selected_prophets,
        'App_Error': APPROXIMATION_ERROR_FULL,
        'Est_Error': mean_estimation_error
    }


# --- SCENARIO 5: Main Function ---
def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    print("\n--- Scenario 5: School of Prophets ---")

    # Load the master pool of high-quality prophets [0, 0.2]
    Y_train_master, True_Risk_master, Prophets_master = load_scenario_5_data('scenario_five_prophets.pkl')

    if Y_train_master is None:
        return

    # Define the parameter grid
    K_VALUES = [2, 5, 10, 50]  # Hypothesis Class Size (k) - Rows
    M_VALUES = [1, 10, 50, 1000]  # Training Sample Size (m) - Columns

    results_by_k = {}
    all_prophet_indices = np.arange(True_Risk_master.size)

    # --- Storage for Plotting ---
    # Store Estimation Error for each k group across all m values
    estimation_errors_for_plot = []

    # --- Nested Loops for all 16 combinations (k x m) ---
    for k in K_VALUES:
        # 1. SAMPLE k PROPHETS (Hypothesis Class) ONCE PER K-ROW
        k_indices = np.random.choice(all_prophet_indices, size=k, replace=False)
        True_Risk_k_subset = True_Risk_master[k_indices]
        Prophets_k_subset = Prophets_master[k_indices, :]  # The fixed k-subset models

        row_results = []
        k_estimation_errors = []  # Temporary list for plot data
        for m in M_VALUES:
            # 2. RUN EXPERIMENTS (100 times) for the fixed k-subset and varying m games
            metrics = run_erm_experiment_5(
                Y_train_master,
                True_Risk_k_subset,
                Prophets_k_subset,
                N_GAMES=m
            )

            # Store data for the plot
            k_estimation_errors.append(metrics['Est_Error'])

            # 3. Format result string for the table
            result_string = f"{metrics['Avg_Error']:.4f} | {metrics['App_Error']:.4f} | {metrics['Est_Error']:.4f}"
            row_results.append(result_string)

        results_by_k[k] = row_results
        estimation_errors_for_plot.append(k_estimation_errors)  # Add all m values for this k

    # --- Output Final Table ---
    header = ["k \\ m"] + [f"m={m}" for m in M_VALUES]
    print("-" * 100)
    print("| " + " | ".join(f"{h:<10}" for h in header) + " |")
    print("|" + "-" * 102 + "|")

    for k, row_data in results_by_k.items():
        row_output = [f"k={k}"] + row_data
        print("| " + " | ".join(f"{d:<10}" for d in row_output) + " |")

    print("-" * 100)
    print("\nKey for each cell: Average Error | Approximation Error | Estimation Error")

    return results_by_k


def load_data_and_prophets_6(prophet_file_name):
    """
    Loads general game data and scenario-specific prophet data using correct keys.
    Returns: Y_train, True_Risk, Prophets_scenario (predictions)
    (Y_train is the true outcome array for training games)
    """
    try:
        # Load general game data
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        Y_train = data['train_set']  # Corrected key for true outcomes
        Y_test = data['test_set']

        # Load scenario-specific prophet data (e.g., Many Prophets)
        with open(prophet_file_name, 'rb') as f:
            scenario_data = pickle.load(f)

        H1_data = scenario_data['hypothesis1']
        H2_data = scenario_data['hypothesis2']

        H1_Prophets = H1_data['train_set']
        H1_True_Risk = np.array(H1_data['true_risk'])
        H2_Prophets = H2_data['train_set']
        H2_True_Risk = np.array(H2_data['true_risk'])

        return Y_train, Y_test, H1_Prophets, H1_True_Risk, H2_Prophets, H2_True_Risk

    except FileNotFoundError:
        print(f"Error: Could not find required file: {prophet_file_name}.")
        return None, None, None, None, None, None
    except KeyError as e:
        print(f"Error: Missing expected key: {e}. Check the file structure.")
        return None, None, None, None, None, None

def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    print("\n--- Scenario 6 Results: Bias-Complexity Tradeoff ---")

    # Load data
    Y_train, Y_test, H1_Prophets, H1_True_Risk, H2_Prophets, H2_True_Risk = \
        load_data_and_prophets_6('scenario_six_prophets.pkl')

    if Y_train is None:
        return

    # --- Setup Constants ---
    NUM_EXPERIMENTS = 100
    N_GAMES = 10  # Training set size is fixed at 10 games

    # Store all Estimation Errors for the histogram plot
    H1_estimation_errors = []
    H2_estimation_errors = []

    # Store True Risks of selected prophets for final averaging
    H1_selected_true_risks = []
    H2_selected_true_risks = []

    # --- Pre-calculate Approximation Errors (Fixed for the class) ---
    H1_APP_ERROR = np.min(H1_True_Risk)  # Approx 0.30
    H2_APP_ERROR = np.min(H2_True_Risk)  # Approx 0.25

    # --- Start Experiment Loop (100 Trials) ---
    for _ in range(NUM_EXPERIMENTS):
        # 1. Sample 10 unique game indices for the training set
        train_game_ids = np.random.choice(len(Y_train), size=N_GAMES, replace=False)
        y_train_games = Y_train[train_game_ids]

        # --- Helper for calculating ERM result for one hypothesis class ---
        def calculate_erm_result(Prophets, True_Risk):
            # 2. Get predictions for the 10 games from the current class
            prophet_predictions = Prophets[:, train_game_ids]

            # 3. Calculate Empirical Risk (ER) on N=10
            losses = (prophet_predictions.astype(int) != y_train_games.astype(int))
            empirical_risks = np.mean(losses, axis=1)

            # 4. Apply ERM: Select prophet based on minimum Empirical Risk
            min_er = np.min(empirical_risks)
            tolerance = 1e-9
            best_prophet_candidates = np.where(empirical_risks <= min_er + tolerance)[0]

            # Break ties uniformly
            selected_prophet_id = np.random.choice(best_prophet_candidates)

            # 5. Get the True Risk of the selected prophet
            return True_Risk[selected_prophet_id]

        # --- Process H1 ---
        L_h1_selected = calculate_erm_result(H1_Prophets, H1_True_Risk)
        H1_selected_true_risks.append(L_h1_selected)
        H1_estimation_errors.append(L_h1_selected - H1_APP_ERROR)

        # --- Process H2 ---
        L_h2_selected = calculate_erm_result(H2_Prophets, H2_True_Risk)
        H2_selected_true_risks.append(L_h2_selected)
        H2_estimation_errors.append(L_h2_selected - H2_APP_ERROR)

    # --- FINAL CALCULATIONS ---

    # H1 Metrics
    H1_AVG_ERROR = np.mean(H1_selected_true_risks)
    H1_EST_ERROR = np.mean(H1_estimation_errors)

    # H2 Metrics
    H2_AVG_ERROR = np.mean(H2_selected_true_risks)
    H2_EST_ERROR = np.mean(H2_estimation_errors)

    # --- Q1 & Q2: Print Results ---
    print("\n| --- COMPARISON SUMMARY --- |")
    print("| Metric                      | Hypothesis 1 (H1) | Hypothesis 2 (H2) |")
    print("|-----------------------------|-------------------|-------------------|")
    print(f"| Approx. Error (Bias)        | {H1_APP_ERROR:.4f}          | {H2_APP_ERROR:.4f}          |")
    print(f"| Avg Test Error (Q1)         | {H1_AVG_ERROR:.4f}          | {H2_AVG_ERROR:.4f}          |")
    print(f"| Avg Estimation Error (Q2)   | {H1_EST_ERROR:.4f}          | {H2_EST_ERROR:.4f}          |")
    print("|-----------------------------|-------------------|-------------------|")

    # --- Q3: Plot Histogram ---

    # Determine common bin range
    all_errors = np.concatenate([H1_estimation_errors, H2_estimation_errors])
    max_error = np.max(all_errors)
    min_error = np.min(all_errors)

    # Define common bins (same number and range for meaningful comparison)
    # Start slightly below min, end slightly above max
    bins = np.linspace(min_error - 0.01, max_error + 0.01, 25)

    plt.figure(figsize=(12, 6))

    # Plot H1
    plt.hist(H1_estimation_errors, bins=bins, alpha=0.6, label='H1 (Small Class: 5 Prophets)', color='skyblue',
             edgecolor='black')

    # Plot H2
    plt.hist(H2_estimation_errors, bins=bins, alpha=0.6, label='H2 (Large Class: 500 Prophets)', color='salmon',
             edgecolor='black')

    plt.axvline(H1_EST_ERROR, color='blue', linestyle='dashed', linewidth=1, label=f'H1 Mean: {H1_EST_ERROR:.4f}')
    plt.axvline(H2_EST_ERROR, color='red', linestyle='dashed', linewidth=1, label=f'H2 Mean: {H2_EST_ERROR:.4f}')

    plt.title('Distribution of Estimation Errors (N=10 Games)')
    plt.xlabel('Estimation Error (L_selected - L_best)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.show()


#PAC
#3
# Fixed parameters
H = 100  # |H|
delta = 0.01
ln_H = np.log(H)
ln_1_over_delta = np.log(1 / delta)

# Range for epsilon (accuracy)
epsilon_values = np.linspace(0.01, 0.2, 50)

# PAC sample size calculation
# m = (1/epsilon) * (ln(|H|) + ln(1/delta))
m_values = (1 / epsilon_values) * (ln_H + ln_1_over_delta)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, m_values, color='darkred', linewidth=3)

# Initial parameter point for reference (epsilon=0.05, m=185)
m_initial = (1 / 0.05) * (ln_H + ln_1_over_delta)
plt.plot(0.05, m_initial, 'o', color='blue', label=f'Initial Point (m={np.ceil(m_initial):.0f})')

plt.title('Required Samples (m) vs. Desired Accuracy (epsilon)')
plt.xlabel('Desired Accuracy (epsilon)')
plt.ylabel('Minimal Samples Required (m)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


#4
# Fixed parameters
H = 100  # |H|
epsilon = 0.05
ln_H = np.log(H)
factor = 1 / epsilon

# Range for delta (confidence level) - Log scale is required
delta_values = np.logspace(-4, -1, 50)

# PAC sample size calculation
# m = (1/epsilon) * (ln(|H|) + ln(1/delta))
m_values = factor * (ln_H + np.log(1 / delta_values))

# Plotting
plt.figure(figsize=(10, 6))
# semilogx uses log scale for x-axis, as required for delta
plt.semilogx(delta_values, m_values, color='darkgreen', linewidth=3)

# Initial parameter point for reference (delta=0.01, m=185)
m_initial = factor * (ln_H + np.log(1 / 0.01))
plt.plot(0.01, m_initial, 'o', color='blue', label=f'Initial Point (m={np.ceil(m_initial):.0f})')

plt.title('Required Samples (m) vs. Confidence Level (delta)')
plt.xlabel('Confidence Level (delta, Log Scale)')
plt.ylabel('Minimal Samples Required (m)')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.show()


if __name__ == '__main__':
    print(f'Scenario 1 Results:')
    Scenario_1()

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()

