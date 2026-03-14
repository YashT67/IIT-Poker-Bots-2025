# IIT Pokerbots 2026: Sneak Peek Hold'em Bot

This repository contains our bot implementation for the **IIT Pokerbots 2026 Competition**. The game played is **Sneak Peek Hold'em**, a custom variant of Heads-Up No-Limit Texas Hold'em featuring a unique post-flop second-price auction mechanic.

## 🃏 The Game: Sneak Peek Hold'em

Sneak Peek Hold'em follows standard No-Limit Texas Hold'em rules with one major twist:
* **The Post-Flop Auction:** Immediately after the flop, both players submit sealed bids.
* **Second-Price Bidding:** The highest bidder wins but only pays the amount of the *lower* bid into the pot.
* **The Reward:** The winner gets to view one of the opponent's private hole cards (chosen uniformly at random). The loser gets no information. 
* **Ties:** Both players pay their bid amount and both get to see one of the opponent's hole cards.

## 🧠 Bot Architecture & Strategy

Our bot (`bot.py`) is built to be highly adaptive, fast, and exploitative. It relies on a combination of pre-computed game theory, dynamic Monte Carlo simulations, and real-time opponent profiling.

### 1. Opponent Modeling (`OpponentModel`)
The bot tracks the opponent's tendencies across the 1000-round match using Exponential Moving Averages (EMA). It records:
* **VPIP (Voluntarily Put In Pot):** Measures pre-flop looseness.
* **Fold Rate:** How often the opponent folds to aggression.
* **Auction Behavior:** Tracks how aggressively the opponent bids and how often they bet post-auction.
* **Post-Flop Aggression:** Adjusts our calling stations based on their bluffing frequencies.

### 2. Adaptive Strategy Engine (`AdaptiveStrategy`)
Instead of static thresholds, the bot dynamically shifts its decision-making boundaries. 
* If the opponent is a "calling station," the bot reduces bluffing and demands higher equity for value bets.
* If the opponent is "tight," the bot lowers its fold threshold to steal more pots.
* It uses sigmoid functions for smooth transitions between strategic states to avoid exploitable discontinuities.

### 3. Hand Evaluation & Equity Calculation
To meet the strict 20-second time limit across 1000 rounds, equity calculation is split:
* **Pre-flop:** Uses a hardcoded, highly optimized lookup table mapping standard hand strings (e.g., `AKs`, `55`) to pre-computed win probabilities.
* **Post-flop:** Utilizes **Monte Carlo Simulations**. It shuffles the remaining deck and generates opponent hands based on their inferred range and any cards revealed during the auction phase. Caching (`@lru_cache`) is used heavily to prevent redundant calculations.

### 4. Board Texture & SPR
* **Board Texture:** Analyzes the community cards for flush draws, straight draws, pairings, and high cards to calculate a "wetness" score. This dictates how aggressively we need to protect our made hands.
* **Stack-to-Pot Ratio (SPR):** Geometric pot commitment calculations drive our all-in and fold logic. A low SPR forces unyielding commit lines, while a high SPR allows for multi-street implied odds plays.

### 5. Auction Bidding Logic
The bot's auction bids are calculated dynamically based on:
* **Position:** Out-of-position (OOP) bids are multiplied by a factor (`OOP_BID_MULTIPLIER = 1.4`) because information is vastly more valuable when acting first.
* **Hand Strength:** Higher equity hands bid more to ensure they have perfect information before committing stacks.
* **Opponent History:** If the opponent over-values auctions, the bot gracefully steps aside or bids just enough to drain their chips.

## ⏱️ Performance & Constraints

The competition engine enforces strict constraints, which this bot is optimized to beat:
* **Query Time Limit:** 2 seconds maximum per query.
* **Total Time Limit:** 20 seconds total across 1000 rounds.
* **Optimization:** By leveraging `eval7` for core hand evaluation, pre-computing pre-flop equities, and using `functools.lru_cache`, the bot minimizes CPU cycles to safely stay within the global timebank.

## Documentation

Refer to **BOT_GUIDE.md** for:
*   Detailed API documentation.
*   Explanation of `PokerState`, `GameInfo`, and `Observation` objects.
*   Available actions and game logic.

## Folder Structure & Imports

To ensure your bot runs correctly, especially regarding imports, you must maintain the following folder structure. The `pkbot` package must be located in the same directory as your `bot.py` file.

```text
.
├── bot.py              # Your bot implementation
├── pkbot/              # Game engine package (do not modify)
├── config.py           # Configuration for the engine
├── engine.py           # The game engine executable
└── requirements.txt    # Python dependencies
```

**Crucial:** Do not move `pkbot` or change the import statements in `bot.py`. The engine relies on `pkbot` being importable as a local package relative to your bot.

## How to Run

0. **Clone this repository**
   Clone this repository into your system to run test matches between bots of your choice.

1. **Install Dependencies:**
   Make sure you have Python 3 installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure the Match:**
   Edit `config.py` to specify which bots to run. You can point to different bot files here.

3. **Start the Engine:**
   Execute the engine script from the root directory:
   ```bash
    python engine.py
   ```
   This will launch the game engine and spawn two instances of the bots defined in `config.py`.

   You can also run with compressed logs using `python engine.py --small_log`.
