import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets

# Generate player inherent MMRs


def generate_inherent_MMRs(n, distribution='normal', mu=50, sigma=15):
    if distribution == 'uniform':
        inherent_MMRs = np.random.uniform(low=1, high=100, size=n)
    elif distribution == 'normal':
        inherent_MMRs = np.random.normal(loc=mu, scale=sigma, size=n)
        inherent_MMRs = np.clip(inherent_MMRs, 1, 100)
    else:
        raise ValueError("Invalid distribution specified")
    return inherent_MMRs


def winning_probability(MMR1, MMR2):
    return 1 / (1 + math.pow(10, ((MMR2 - MMR1) / 400)))


# ELO rating update


def update_elo(r1, r2, outcome, k=32):
    e1 = 1 / (1 + 10**((r2 - r1) / 400))
    e2 = 1 / (1 + 10**((r1 - r2) / 400))
    r1_new = r1 + k * (outcome - e1)
    r2_new = r2 + k * ((1 - outcome) - e2)
    return r1_new, r2_new

# Simulate a round of pairwise combats


def simulate_combats(inherent_MMRs, game_MMRs, map_size, dead_players, kill_counter):
    alive_players = [i for i in range(len(game_MMRs)) if i not in dead_players]
    random.shuffle(alive_players)

    previous_combat_game_MMRs = {}
    k = 0.1

    for i in range(0, len(alive_players), 2):
        if i + 1 < len(alive_players):
            player1 = alive_players[i]
            player2 = alive_players[i + 1]

            MMR_diff = inherent_MMRs[player1] - inherent_MMRs[player2]
            win_probability = 1 / (1 + math.exp(-k * MMR_diff))

            if random.random() < win_probability:
                outcome = 1
            else:
                outcome = 0

            game_MMRs[player1], game_MMRs[player2] = update_elo(
                game_MMRs[player1], game_MMRs[player2], outcome)

            if outcome:
                dead_players.add(player2)
                kill_counter[player1] = kill_counter.get(player1, 0) + 1
                previous_combat_game_MMRs[player1] = game_MMRs[player2]
            else:
                dead_players.add(player1)
                kill_counter[player2] = kill_counter.get(player2, 0) + 1
                previous_combat_game_MMRs[player2] = game_MMRs[player1]

    return game_MMRs, dead_players, previous_combat_game_MMRs


# Generate the map and plot it


def plot_map(game_MMRs, dead_players, round_number):
    map_size = int(np.sqrt(len(game_MMRs)))
    map_data = np.zeros((map_size, map_size))
    for i in range(len(game_MMRs)):
        x = i % map_size
        y = i // map_size
        if i in dead_players:
            map_data[y, x] = 0
        else:
            map_data[y, x] = game_MMRs[i]

    plt.imshow(map_data, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Round {round_number}")
    plt.show()

# Generate the table and print it


def print_table(game_MMRs, game_MMRs_initial, inherent_MMRs, dead_players, kill_counter, previous_combat_game_MMRs):
    data = []
    for i in range(len(game_MMRs)):
        status = "Dead" if i in dead_players else "Alive"
        total_MMR_gain = game_MMRs[i] - game_MMRs_initial[i]
        kills = kill_counter.get(i, 0)
        if status == "Alive" and i in previous_combat_game_MMRs:
            prev_combat_opponent_MMR = previous_combat_game_MMRs[i]
            win_probability = winning_probability(
                inherent_MMRs[i], prev_combat_opponent_MMR)
        else:
            win_probability = None
        last_opponent_game_MMR = previous_combat_game_MMRs.get(i, "-")

        data.append([i + 1, inherent_MMRs[i], game_MMRs[i],
                    total_MMR_gain, kills, status, win_probability, last_opponent_game_MMR])

    df = pd.DataFrame(data, columns=['Player', 'Inherent MMR', 'Current MMR',
                      'Total MMR Gain', 'Kills', 'Status', 'Win Probability', 'Last Opponent Game MMR'])
    df = df.sort_values(
        by='Current MMR', ascending=False).reset_index(drop=True)
    display(df)


# Main program


def main(n_rounds):
    n_players = 100
    distribution = 'normal'
    mu = 50
    sigma = 15
    map_size = 10
    kill_counter = {}

    inherent_MMRs = generate_inherent_MMRs(n_players, distribution, mu, sigma)
    game_MMRs = np.copy(inherent_MMRs)
    game_MMRs_initial = np.copy(game_MMRs)
    dead_players = set()

    for i in range(n_rounds):
        game_MMRs, dead_players, previous_combat_game_MMRs = simulate_combats(
            inherent_MMRs, game_MMRs, map_size, dead_players, kill_counter)
        plot_map(game_MMRs, dead_players, i + 1)
        print_table(game_MMRs, game_MMRs_initial, inherent_MMRs,
                    dead_players, kill_counter, previous_combat_game_MMRs)


def on_button_click(button):
    global n_rounds
    n_rounds += 1
    clear_output(wait=True)
    display(button)
    main(n_rounds)


# Display the button and set the callback
button = widgets.Button(description="Simulate Next Round")
button.on_click(on_button_click)
display(button)

# Run the main program with 1 round initially
n_rounds = 1
main(n_rounds)
