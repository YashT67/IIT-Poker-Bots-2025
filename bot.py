"""
bot.py — Heads-Up No-Limit Hold'em Poker Bot with Auction Mechanic Support

This bot plays heads-up (2-player) poker. It uses:
  - Pre-computed pre-flop equity tables for fast pre-flop decisions
  - Monte Carlo simulation for post-flop equity estimation
  - An opponent model that tracks aggression, looseness, and auction behavior
  - An adaptive strategy layer that shifts thresholds based on the opponent's profile
  - A Vickrey-style auction bidding engine to reveal one of the opponent's hole cards
  - Board texture analysis to differentiate between wet/dry boards
  - Stack-to-Pot Ratio (SPR) logic for commitment-level decisions

Entry point: `Player.get_move()` is called each time the bot must act.
"""

import random
import math
import eval7
from functools import lru_cache
from collections import deque

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot


# =============================================================================
# SECTION 1: GLOBAL BEHAVIORAL CONSTANTS
#
# These constants control the bot's strategy at a high level.
# Changing these values will affect aggression, bidding, bluffing frequency,
# and how the bot responds to various game situations without modifying logic.
# =============================================================================

# When the bot is Out Of Position (i.e., it is the Big Blind and acts first
# post-flop), information from the auction is more valuable because the
# opponent will always have positional advantage. This multiplier increases
# the raw computed bid by 40% to compensate.
OOP_BID_MULTIPLIER = 1.4

# Number of Monte Carlo simulation iterations used for post-flop equity
# estimation. Higher values = more accurate but slower. 1000 is a good
# balance for real-time play.
MC_SIMS = 1600

# Separate (lower) simulation count used during the auction phase only.
# Kept lower because compute_bid() already runs multiple equity calculations
# in parallel, and we need a response quickly.
MC_SIMS_AUCTION = 200

# Equity thresholds that gate different actions:
#   FOLD_THRESHOLD  — below this, prefer folding when facing a bet
#   CALL_THRESHOLD  — minimum equity needed to call a bet
#   VALUE_THRESHOLD — minimum equity required to fire a value bet/raise
FOLD_THRESHOLD  = 0.35
CALL_THRESHOLD  = 0.42
VALUE_THRESHOLD = 0.66

# Probability (0.0–1.0) that the bot attempts a bluff when it would otherwise
# check. Applied stochastically so bluffing isn't predictable.
BLUFF_FREQ = 0.16

# Default bet sizing fractions relative to the pot.
# These are starting points; they get adjusted by street and board texture.
BET_SMALL_FRAC = 0.50   # Half-pot sizing for thin value / merged ranges
BET_BIG_FRAC   = 0.75   # Three-quarter pot for strong value hands

# Minimum hand history before the opponent model's statistics are trusted.
# Before this many hands, all thresholds fall back to baseline defaults.
MODEL_MIN_HANDS = 40

# Stack-to-Pot Ratio (SPR) boundary values:
#   SPR < SPR_LOW  → players are pot-committed, simplify to jam/fold
#   SPR > SPR_HIGH → deep stacks, drawing hands gain implied odds value
SPR_LOW  = 3.0
SPR_HIGH = 9.0

# Range fractions map to the top X% of all possible starting hands,
# sorted by pre-flop equity. Used to constrain MC opponent hand sampling.
RANGE_TIGHT  = 0.30   # Only the top 30% of hands (tight/nit opponent)
RANGE_MEDIUM = 0.60   # Top 60% of hands (average opponent)
RANGE_LOOSE  = 0.96   # All hands (no range constraint applied)

# Integer rank thresholds used when interpreting a card revealed during the
# auction phase. Rank values: 2=2, ..., 10=T, 11=J, 12=Q, 13=K, 14=A.
REVEALED_HIGH_RANK = 11   # Jack or above → opponent has a strong card
REVEALED_LOW_RANK  = 7    # 7 or below → opponent has a weak card

# When the opponent's revealed auction card is high (Jack+):
#   - Reduce our equity estimate slightly (opponent has a good card)
#   - Reduce bluff frequency (harder to fold out a strong hand)
#   - Reduce bet sizing (we lose value betting into a good range)
REVEALED_HIGH_EQ_DELTA    = -0.04
REVEALED_HIGH_BLUFF_SCALE =  0.70
REVEALED_HIGH_SIZE_SCALE  =  0.85

# When the revealed card is low (7 or below):
#   - Increase equity slightly (opponent has a weak card)
#   - Increase bluff frequency (easier to fold them out)
#   - Increase bet sizing (we get more value)
REVEALED_LOW_EQ_DELTA    = 0.02
REVEALED_LOW_BLUFF_SCALE = 1.10
REVEALED_LOW_SIZE_SCALE  = 1.15

# Dynamic ceiling for how large our auction bid can be as a fraction of the
# pot. The range [BID_CAP_MIN, BID_CAP_MAX] is scaled based on pot size and
# stack depth inside dynamic_max_bid_frac().
BID_CAP_MIN = 0.16   # Never bid below 16% of pot
BID_CAP_MAX = 0.60   # Never bid above 60% of pot

# When the opponent wins the auction (they see one of their cards), the bot
# plays more conservatively. AUCTION_LOSS_K is a multiplier close to 1.0 that
# gently tightens bluff frequency and shifts equity thresholds upward.
# Value of 0.84 means a 16% tightening effect.
AUCTION_LOSS_K = 0.84

# Minimum auction wins before we fully trust revealed card adjustments.
# Below this, adjustments are dampened proportionally.
AUCTION_CONFIDENCE_MIN = 16

# Thresholds for interpreting the opponent's flop-betting behavior after they
# win the auction. A high post-auction bet rate signals that the revealed card
# genuinely strengthened their hand or they are very aggressive.
OPP_AUCT_AGG_HIGH  = 0.65   # ≥65% → opponent almost always bets with auction card
OPP_AUCT_AGG_LOW   = 0.30   # ≤30% → opponent often traps or slow-plays
OPP_AUCT_FOLD_BUMP = 0.07   # Raise fold threshold by this much vs. aggressive post-auction bettors
OPP_AUCT_CALL_DIP  = 0.05   # Lower call threshold by this much vs. passive post-auction players

# Equity and bluff adjustments based on the bot's table position.
# In position (SB / not BB): slight equity bonus since we act last post-flop.
# Out of position (BB): slight penalty since we are at an informational disadvantage.
POSITION_IP_EQ_BONUS    =  0.02   # Add to equity when in position (SB)
POSITION_OOP_EQ_DELTA   = -0.02   # Subtract from equity when out of position (BB)
POSITION_IP_BLUFF_SCALE =  1.10   # Multiply bluff frequency by 1.1 when in position
POSITION_OOP_BLUFF_SCALE =  0.90   # Multiply bluff frequency by 0.9 when out of position
POSITION_BB_CALL_BONUS  =  0.04   # Lower the call threshold by 0.04 for BB (close the action discount)

# When facing an aggressive opponent with a strong hand, occasionally check
# to induce a bet and then reraise (check-raise trap).
CHECK_RAISE_FREQ    = 0.28   # Probability of trapping with a strong hand vs aggro opponent
CHECK_RAISE_MIN_AGG = 0.40   # Minimum opponent postflop_aggression to trigger trap

# River-specific polarized sizing: bet big with value and bluffs (GTO polarization).
RIVER_VALUE_SIZE      = 0.85  # Pot fraction for river value bets (polarized large)
RIVER_BLUFF_REDUCTION = 0.60  # Multiply bluff frequency by this on the river


# =============================================================================
# SECTION 2: CARD UTILITY PRIMITIVES
#
# Basic helpers for parsing card strings like 'Ah', 'Td', '2c'.
# =============================================================================

# All valid card ranks in ascending order (2 through Ace).
RANKS = '23456789TJQKA'

# All four suits in no particular order.
SUITS = 'hdcs'

# Maps rank characters to integer values (e.g., '2'→2, 'T'→10, 'A'→14).
# enumerate(..., 2) starts the counter at 2 so '2'→2 naturally.
RANK_VAL = {r: i for i, r in enumerate(RANKS, 2)}

# Reverse mapping: integer rank -> display character. Used by _hand_key().
_RD = {v: k for k, v in RANK_VAL.items()}

# Complete 52-card deck as a flat list of strings, e.g., ['2h','2d',...,'As','Ac'].
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def card_rank(card: str) -> int:
    """
    Extract the integer rank value from a card string.

    Example:
        card_rank('Ah') → 14
        card_rank('Tc') → 10
        card_rank('2d') →  2
    """
    return RANK_VAL[card[0]]


def card_suit(card: str) -> str:
    """
    Extract the suit character from a card string.

    Example:
        card_suit('Ah') → 'h'
        card_suit('2c') → 'c'
    """
    return card[1]


# =============================================================================
# SECTION 3: HAND EVALUATION WRAPPERS (eval7 library)
#
# eval7 is a fast Cython-based poker hand evaluator.
# Higher return values mean stronger hands.
# =============================================================================

def convert_to_cards(cards):
    """
    Convert a list of card strings (e.g., ['Ah', 'Kd']) into eval7.Card objects.
    eval7 requires its own Card type for hand evaluation.
    """
    return [eval7.Card(str(c)) for c in cards]


# Module-level card-string -> eval7.Card cache. Avoids repeated object creation
# inside hot MC loops. There are only 52 possible card strings.
_E7_CACHE: dict[str, eval7.Card] = {}


def _e7(card_str: str) -> eval7.Card:
    """Return the cached eval7.Card for a given card string like 'Ah'."""
    c = _E7_CACHE.get(card_str)
    if c is None:
        c = eval7.Card(card_str)
        _E7_CACHE[card_str] = c
    return c


# Pre-populate the cache for all 52 cards at import time.
for _cs in FULL_DECK:
    _e7(_cs)


def best_hand_score(hole: list, board: list) -> int:
    """
    Evaluate the best possible 5-card hand from a player's hole cards and
    any number of community cards (can be 3, 4, or 5 board cards).

    eval7.evaluate() automatically selects the best 5-card combination
    from all cards provided.

    Args:
        hole:  The player's 2 private hole cards.
        board: The community cards visible to all players.

    Returns:
        Integer strength score (higher = better).
    """
    all_cards = convert_to_cards(hole + board)
    return eval7.evaluate(all_cards)


# =============================================================================
# SECTION 4: PRE-FLOP EQUITY LOOKUP TABLE
#
# Running Monte Carlo simulations pre-flop on every action would be too slow
# and is unnecessary since pre-flop equities are static. This table stores
# the win probability for every canonical heads-up starting hand versus a
# random opponent hand.
#
# Keys:
#   'AA', 'KK', ...         → Pocket pairs (always 2 chars)
#   'AKs', 'QTs', ...       → Suited unpaired hands ('s' suffix)
#   'AKo', 'Q8o', ...       → Offsuit unpaired hands ('o' suffix)
# Values:
#   Float representing win probability (e.g., 0.853 = 85.3% vs. random hand)
# =============================================================================

_PF_EQUITY: dict[str, float] = {
    # Pocket Pairs
    'AA': 0.852, 'KK': 0.824, 'QQ': 0.799, 'JJ': 0.775, 'TT': 0.751,
    '99': 0.716, '88': 0.689, '77': 0.662, '66': 0.633, '55': 0.603,
    '44': 0.570, '33': 0.537, '22': 0.503,

    # Suited Aces
    'AKs': 0.670, 'AQs': 0.661, 'AJs': 0.654, 'ATs': 0.647, 'A9s': 0.628,
    'A8s': 0.619, 'A7s': 0.609, 'A6s': 0.599, 'A5s': 0.599, 'A4s': 0.588,
    'A3s': 0.579, 'A2s': 0.569,

    # Offsuit Aces
    'AKo': 0.653, 'AQo': 0.644, 'AJo': 0.635, 'ATo': 0.627, 'A9o': 0.607,
    'A8o': 0.596, 'A7o': 0.584, 'A6o': 0.573, 'A5o': 0.573, 'A4o': 0.561,
    'A3o': 0.551, 'A2o': 0.540,

    # Suited Kings
    'KQs': 0.633, 'KJs': 0.625, 'KTs': 0.617, 'K9s': 0.599, 'K8s': 0.584,
    'K7s': 0.576, 'K6s': 0.567, 'K5s': 0.557, 'K4s': 0.547, 'K3s': 0.537,
    'K2s': 0.528,

    # Offsuit Kings
    'KQo': 0.614, 'KJo': 0.606, 'KTo': 0.597, 'K9o': 0.577, 'K8o': 0.560,
    'K7o': 0.551, 'K6o': 0.541, 'K5o': 0.530, 'K4o': 0.518, 'K3o': 0.507,
    'K2o': 0.497,

    # Suited Queens
    'QJs': 0.602, 'QTs': 0.594, 'Q9s': 0.575, 'Q8s': 0.559, 'Q7s': 0.545,
    'Q6s': 0.536, 'Q5s': 0.525, 'Q4s': 0.514, 'Q3s': 0.504, 'Q2s': 0.495,

    # Offsuit Queens
    'QJo': 0.582, 'QTo': 0.573, 'Q9o': 0.552, 'Q8o': 0.535, 'Q7o': 0.519,
    'Q6o': 0.508, 'Q5o': 0.497, 'Q4o': 0.484, 'Q3o': 0.473, 'Q2o': 0.463,

    # Suited Jacks
    'JTs': 0.575, 'J9s': 0.555, 'J8s': 0.539, 'J7s': 0.525, 'J6s': 0.508,
    'J5s': 0.497, 'J4s': 0.485, 'J3s': 0.475, 'J2s': 0.465,

    # Offsuit Jacks
    'JTo': 0.553, 'J9o': 0.531, 'J8o': 0.513, 'J7o': 0.497, 'J6o': 0.479,
    'J5o': 0.466, 'J4o': 0.453, 'J3o': 0.442, 'J2o': 0.431,

    # Suited Tens
    'T9s': 0.540, 'T8s': 0.523, 'T7s': 0.508, 'T6s': 0.491, 'T5s': 0.474,
    'T4s': 0.461, 'T3s': 0.450, 'T2s': 0.440,

    # Offsuit Tens
    'T9o': 0.516, 'T8o': 0.497, 'T7o': 0.480, 'T6o': 0.461, 'T5o': 0.442,
    'T4o': 0.427, 'T3o': 0.415, 'T2o': 0.403,

    # Suited Nines
    '98s': 0.508, '97s': 0.492, '96s': 0.474, '95s': 0.459, '94s': 0.441,
    '93s': 0.430, '92s': 0.419,

    # Offsuit Nines
    '98o': 0.481, '97o': 0.463, '96o': 0.443, '95o': 0.426, '94o': 0.405,
    '93o': 0.393, '92o': 0.380,

    # Suited Eights
    '87s': 0.479, '86s': 0.461, '85s': 0.445, '84s': 0.428, '83s': 0.410,
    '82s': 0.399,

    # Offsuit Eights
    '87o': 0.450, '86o': 0.430, '85o': 0.412, '84o': 0.392, '83o': 0.372,
    '82o': 0.358,

    # Suited Sevens
    '76s': 0.457, '75s': 0.440, '74s': 0.423, '73s': 0.404, '72s': 0.385,

    # Offsuit Sevens
    '76o': 0.426, '75o': 0.407, '74o': 0.387, '73o': 0.366, '72o': 0.346,

    # Suited Sixes
    '65s': 0.438, '64s': 0.421, '63s': 0.402, '62s': 0.381,

    # Offsuit Sixes
    '65o': 0.406, '64o': 0.386, '63o': 0.365, '62o': 0.342,

    # Suited Fives
    '54s': 0.427, '53s': 0.409, '52s': 0.389,

    # Offsuit Fives
    '54o': 0.394, '53o': 0.373, '52o': 0.351,

    # Suited Fours
    '43s': 0.400, '42s': 0.380,

    # Offsuit Fours
    '43o': 0.364, '42o': 0.341,

    # Suited Threes
    '32s': 0.365,

    # Offsuit Threes
    '32o': 0.323
}


@lru_cache(maxsize=2704)
def _hand_key(c1: str, c2: str) -> str:
    """
    Normalize two hole cards into a canonical lookup key for _PF_EQUITY.

    The key format follows standard poker shorthand:
        - Pocket pairs:     'AA', 'KK', '22', etc.
        - Suited hands:     'AKs', 'QTs', etc.  (higher rank first)
        - Offsuit hands:    'AKo', 'J9o', etc.  (higher rank first)

    Uses lru_cache because the same card pair can appear hundreds of times
    across a long session, and string construction is surprisingly expensive.

    Args:
        c1, c2: Two card strings, e.g., 'Ah', 'Kd'

    Returns:
        A string key that matches a key in _PF_EQUITY.
    """
    r1, r2 = card_rank(c1), card_rank(c2)
    s1, s2 = card_suit(c1), card_suit(c2)

    # Always put the higher-rank card first in the key string.
    hi_r, lo_r = max(r1, r2), min(r1, r2)

    # Pocket pairs: same rank, no suit suffix needed.
    if hi_r == lo_r:
        return _RD[hi_r] + _RD[lo_r]

    # Unpaired hands: append 's' for suited, 'o' for offsuit.
    suffix = 's' if s1 == s2 else 'o'
    return _RD[hi_r] + _RD[lo_r] + suffix


def preflop_equity(hole: list) -> float:
    """
    Return the pre-flop win probability for our hole cards against a random
    opponent hand in heads-up play.

    Primary path: direct lookup in _PF_EQUITY via the canonical key.

    Default fallback: if the key is somehow missing (should not happen), return 0.5 as a neutral equity guess.
    
    Args:
        hole: List of exactly 2 card strings.

    Returns:
        Float in [0.0, 1.0] representing our win probability pre-flop.
    """
    key = _hand_key(hole[0], hole[1])
    if key in _PF_EQUITY:
        return _PF_EQUITY[key]

    return 0.5  # Default fallback equity if key is missing (should not occur)

# =============================================================================
# SECTION 5: OPPONENT RANGE CONSTRUCTION
#
# We constrain the Monte Carlo simulations so that the opponent is only ever
# assigned hands from within their estimated "range" (the set of hands they
# would actually play). This makes equity estimates more realistic than
# simulating vs. all 52 cards randomly.
# =============================================================================

# Pre-sort all hand keys from strongest to weakest pre-flop equity.
# This is the "range stack": the top N% of this list forms an opponent's range.
_ALL_HAND_KEYS_SORTED: list[str] = sorted(
    _PF_EQUITY.keys(), key=lambda k: _PF_EQUITY[k], reverse=True
)

# Pre-build the mapping from canonical hand key → all physical card pairs
# for that hand class. For example:
#   'AA' → [('Ah','Ad'), ('Ah','Ac'), ('Ah','As'), ('Ad','Ac'), ...]
#   'AKs'→ [('Ah','Kh'), ('Ad','Kd'), ('Ac','Kc'), ('As','Ks')]
#   'AKo'→ all 12 offsuit combinations of an Ace and a King
#
# Computing this at import time avoids repeated work during simulations.
_CARDS_FOR_KEY_FULL: dict[str, list] = {}

for _key in _ALL_HAND_KEYS_SORTED:
    if len(_key) == 2:
        # Pocket pair: both cards share the same rank, different suits.
        # We use _s1 < _s2 to avoid duplicate pairs like ('Ah','Ad') and ('Ad','Ah').
        _r = _key[0]
        _CARDS_FOR_KEY_FULL[_key] = [(_r + _s1, _r + _s2)
                                      for _s1 in SUITS for _s2 in SUITS
                                      if _s1 < _s2]
    else:
        # Unpaired hand: separate high rank, low rank, and suited flag.
        _hi_r, _lo_r, _suited = _key[0], _key[1], _key[2] == 's'
        _pairs = []
        for _s1 in SUITS:
            for _s2 in SUITS:
                # For suited hands: both cards must share a suit.
                if _suited and _s1 != _s2:
                    continue
                # For offsuit hands: suits must differ.
                if not _suited and _s1 == _s2:
                    continue
                _c1, _c2 = _hi_r + _s1, _lo_r + _s2
                # Sanity guard: a card cannot be paired with itself.
                if _c1 != _c2:
                    _pairs.append((_c1, _c2))
        _CARDS_FOR_KEY_FULL[_key] = _pairs


def _cards_for_key(key: str, excluded: set) -> list:
    """
    Return all physical card pairs for a given hand class that do not contain
    any cards already in use (in our hand, on the board, or in opp_known).

    Args:
        key:      Canonical hand key like 'AKs', 'QQ'.
        excluded: Set of card strings that cannot be dealt.

    Returns:
        List of (card1, card2) tuples still available to be assigned.
    """
    return [(a, b) for a, b in _CARDS_FOR_KEY_FULL[key]
            if a not in excluded and b not in excluded]


@lru_cache(maxsize=256)
def _build_opponent_range_cached(range_fraction: float,
                                  excluded_frozen: frozenset) -> tuple:
    """
    Build and cache the set of valid opponent hand pairs for a given range
    fraction and set of excluded (dead) cards.

    The range is constructed by taking the top `range_fraction * N` hand
    types from _ALL_HAND_KEYS_SORTED and expanding each to concrete card pairs,
    filtering out dead cards.

    Args:
        range_fraction:   Float in (0, 1.0] representing the fraction of all
                          hand types to include (e.g., 0.25 = top 25%).
        excluded_frozen:  frozenset of unavailable card strings (must be
                          hashable for lru_cache).

    Returns:
        Tuple of (card1, card2) pairs making up the opponent's playable range.
    """
    n_types  = max(1, int(len(_ALL_HAND_KEYS_SORTED) * range_fraction))
    top_keys = _ALL_HAND_KEYS_SORTED[:n_types]
    pairs    = []
    for key in top_keys:
        pairs.extend(_cards_for_key(key, excluded_frozen))
    return tuple(pairs)


def build_opponent_range(range_fraction: float, excluded: set) -> list:
    """
    Public wrapper around _build_opponent_range_cached that accepts a regular
    set (which is not hashable) by converting it to a frozenset internally.

    Args:
        range_fraction: Fraction of all hand types to include.
        excluded:       Set of dead card strings.

    Returns:
        List of valid (card1, card2) opponent hand pairs.
    """
    return list(_build_opponent_range_cached(range_fraction, frozenset(excluded)))


# =============================================================================
# SECTION 6: MONTE CARLO EQUITY ESTIMATION
#
# For post-flop decisions we cannot rely on static tables — the board changes
# the relative strength of every hand. Instead, we run N random "runouts"
# (deal out the remaining board cards and a random opponent hand) and count
# how often our hand wins.
#
# This section handles two cases:
#   Case 1: One opponent card is known; the other is sampled from their range.
#   Case 2: Neither opponent card is known; both drawn randomly from range.
# =============================================================================

def monte_carlo_equity(hole: list, board: list, opp_known: list,
                       n_sims: int = MC_SIMS,
                       opp_range_fraction: float = RANGE_LOOSE) -> float:
    """
    Estimate our win probability via Monte Carlo simulation.

    For each simulation iteration:
        1. Sample a valid opponent hand (respecting known cards and their range).
        2. Complete the community board to 5 cards.
        3. Evaluate both hands and record win/tie/loss.
    Returns the average win rate (ties count as 0.5 wins).

    Args:
        hole:               Our 2 hole cards.
        board:              Current community cards (0–5 cards).
        opp_known:          Cards we know the opponent holds (0, 1, or 2).
                            Usually 1 card if we won the auction.
        n_sims:             Number of Monte Carlo iterations.
        opp_range_fraction: Fraction of all hand types the opponent plays.
                            Lower = tighter range = stronger opponent hands on average.

    Returns:
        Float in [0.0, 1.0] representing our estimated win probability.
    """
    # Build the set of all cards that are already "in play" and cannot be dealt.
    known_set = set(hole + board + opp_known)
    remaining = [c for c in FULL_DECK if c not in known_set]

    # Determine how many additional cards we need to complete the board and
    # how many of the opponent's hole cards are still unknown.
    cards_needed_board = 5 - len(board)
    cards_needed_opp   = 2 - len(opp_known)

    # Pre-build the opponent range if their hand isn't fully known and we're
    # not using a completely unconstrained (random) range.
    range_pairs = build_opponent_range(opp_range_fraction, known_set)
    second_card_pool = None  # Will hold the valid second card options if one opponent card is known.
    if cards_needed_opp == 1:
        # If one opponent card is known, filter the range to only pairs that include that card.
        known_card = opp_known[0]
        second_card_pool = list({
                c
                for pair in range_pairs
                for c in pair
                if c != known_card and c not in known_set
            })

    # Pre-convert known cards to eval7 objects ONCE before the loop.
    e7_hole = [_e7(c) for c in hole]
    e7_board = [_e7(c) for c in board]
    _eval = eval7.evaluate   # Local reference avoids attribute lookups

    wins = 0.0
    _sample = random.sample   # Local reference avoids repeated global dict lookups
    _choice = random.choice

    for _ in range(n_sims):

        # ----- CASE 1: One opponent card is known; sample the other -----
        if second_card_pool is not None:
            second = _choice(second_card_pool)
            opp_hole_strs = [opp_known[0], second]
            e7_opp = [_e7(c) for c in opp_hole_strs]
            pool = [c for c in remaining if c != second]

        # ----- CASE 2: No opponent cards known -----
        else:
            pair = _choice(range_pairs)
            opp_hole_strs = list(opp_known) + [c for c in pair
                                                if c not in opp_known][:cards_needed_opp]
            e7_opp = [_e7(c) for c in opp_hole_strs]
            pool = [c for c in remaining if c not in opp_hole_strs]

        # Complete the board by sampling from the valid pool (Cases 1 and 2).
        sampled = _sample(pool, cards_needed_board)
        e7_sampled = [_e7(c) for c in sampled]

        my  = _eval(e7_hole + e7_board + e7_sampled)
        opp = _eval(e7_opp  + e7_board + e7_sampled)

        if my > opp:
            wins += 1.0
        elif my == opp:
            wins += 0.5

    # Guard against divide-by-zero if n_sims was 0 (should never occur normally).
    return wins / n_sims if n_sims > 0 else 0.5


def get_equity(state: PokerState,
               n_sims: int = MC_SIMS,
               opp_range_fraction: float = RANGE_LOOSE) -> float:
    """
    Top-level equity dispatcher: routes to the appropriate equity function
    depending on the current street.

    - Pre-flop: use the fast lookup table (preflop_equity).
    - All other streets: use Monte Carlo simulation (monte_carlo_equity).

    Args:
        state:              Current game state (provides hole, board, opp cards).
        n_sims:             Number of MC iterations (ignored pre-flop).
        opp_range_fraction: Opponent range constraint passed to MC.

    Returns:
        Float in [0.0, 1.0] — our estimated equity for this hand/board.
    """
    hole  = state.my_hand
    board = list(state.board)              if state.board              else []
    opp   = list(state.opp_revealed_cards) if state.opp_revealed_cards else []

    # Cannot estimate equity without hole cards.
    if not hole or len(hole) < 2:
        return 0.5

    if state.street == 'pre-flop':
        return preflop_equity(hole)

    return monte_carlo_equity(hole, board, opp, n_sims, opp_range_fraction)


# =============================================================================
# SECTION 7: BOARD TEXTURE ANALYSIS
#
# Board texture describes how "dangerous" or "coordinated" the community cards
# are. A "wet" board (many flush/straight draws) requires different bet sizing
# and bluffing strategy than a "dry" board (uncoordinated, no obvious draws).
# =============================================================================

class BoardTexture:
    """
    Analyzes the community cards and produces a suite of boolean flags and a
    continuous 'wetness' score for strategic use.

    Attributes:
        flush_draw   (bool):  Three or more cards share a suit → flush is possible.
        straight_draw(bool):  Three board cards span within 4 ranks → straight draw possible.
        paired       (bool):  At least two board cards share the same rank.
        trips_on_board(bool): Three board cards share the same rank.
        high_board   (bool):  Average board rank is above 10 (face-card heavy).
        wetness     (float):  Aggregate danger score in [0.0, ~1.0].
        danger      (float):  Alias for wetness (provided for semantic clarity).
    """

    def __init__(self, board: list):
        """
        Compute all texture flags and the wetness score from the board cards.

        Args:
            board: List of card strings on the community board (0–5 cards).
        """
        self.board = board
        ranks = [card_rank(c) for c in board]
        suits = [card_suit(c) for c in board]

        # Flush draw: if any suit appears 3+ times, a flush is possible.
        self.flush_draw = max(suits.count(s) for s in SUITS) >= 3 if board else False

        # Straight draw: delegated to the helper below.
        self.straight_draw = self._has_straight_draw(ranks)

        # Paired board: if any rank repeats (but not all cards are unique).
        self.paired = len(ranks) != len(set(ranks)) and len(board) >= 2

        # Trips on board: a full three-of-a-kind exists on the community cards.
        self.trips_on_board = any(ranks.count(r) >= 3 for r in set(ranks))

        # High board: the average rank is above 10 (mostly face cards).
        self.high_board = (sum(ranks) / max(len(ranks), 1)) > 10 if board else False

        # Wetness is a weighted sum of the above flags, calibrated so that a
        # board with both a flush draw and straight draw maxes out near 0.75+.
        # Weights reflect the relative strategic importance of each factor:
        #   Flush draws (0.40) are slightly more dangerous than straight draws (0.35)
        #   because they are harder to spot and block.
        self.wetness = (
            0.40 * int(self.flush_draw)
          + 0.35 * int(self.straight_draw)
          + 0.15 * int(self.paired)
          + 0.10 * int(self.high_board)
        )

    @staticmethod
    def _has_straight_draw(ranks: list) -> bool:
        """
        Detect whether three or more board cards fall within a 5-card window,
        meaning at least one straight draw exists.

        Algorithm:
            1. Deduplicate and sort the ranks.
            2. Check every window of 3 consecutive unique ranks.
               If the highest minus the lowest ≤ 4, they can all be part of
               the same 5-card straight span.
            3. Also check for a "wheel" straight draw (A-2-3-4-5) by treating
               the Ace as rank 1.

        Args:
            ranks: List of integer rank values from the board cards.

        Returns:
            True if a straight draw exists, False otherwise.
        """
        if len(ranks) < 3:
            return False   # Need at least 3 cards to form a meaningful draw

        uniq = sorted(set(ranks))

        # Check every 3-card window in sorted unique ranks.
        for i in range(len(uniq) - 2):
            if uniq[i + 2] - uniq[i] <= 4:
                return True

        # Special case: Ace can play low (as 1) in a wheel straight A-2-3-4-5.
        if 14 in uniq:
            low_uniq = sorted({1 if r == 14 else r for r in uniq})
            for i in range(len(low_uniq) - 2):
                if low_uniq[i + 2] - low_uniq[i] <= 4:
                    return True

        return False

    def __str__(self):
        """
        Human-readable summary of the board texture, useful for debugging.
        Example output: 'Board(FD|SD, wet=0.75)'
        """
        parts = []
        if self.flush_draw:     parts.append('FD')
        if self.straight_draw:  parts.append('SD')
        if self.paired:         parts.append('Paired')
        if self.high_board:     parts.append('High')
        return f'Board({"|".join(parts) or "Dry"}, wet={self.wetness:.2f})'


def board_texture(board: list) -> BoardTexture:
    """
    Convenience factory function: instantiate a BoardTexture from a list of cards.

    Args:
        board: Community card strings.

    Returns:
        A fully initialized BoardTexture instance.
    """
    return BoardTexture(board)


# =============================================================================
# SECTION 8: STACK-TO-POT RATIO (SPR) UTILITIES
#
# SPR = effective stack size / pot size
#
# SPR tells us how many "pot-sized bets" are left behind. Low SPR means the
# money is already committed and we should rarely fold. High SPR means there
# is plenty of room to maneuver — draws become more valuable and bluffs can
# yield bigger folds.
# =============================================================================

def compute_spr(state: PokerState) -> float:
    """
    Compute the Stack-to-Pot Ratio using the smaller of the two players'
    remaining stacks (the "effective stack").

    A high SPR (>8) → deep-stacked, implied odds matter.
    A low SPR (<3)  → pot-committed, lean toward jam/fold.

    Args:
        state: Current game state with chip counts and pot size.

    Returns:
        Float representing SPR. Returns 20.0 if the pot is zero (pre-bet).
    """
    eff_stack = min(state.my_chips, getattr(state, 'opp_chips', state.my_chips))
    return eff_stack / state.pot if state.pot > 0 else 20.0


def spr_adjustment(spr: float, equity: float) -> float:
    """
    Nudge our effective equity estimate up or down based on the SPR.

    Low SPR (committed pot):
        - If we are already ahead (equity ≥ 55%), push further toward commit (+0.05).
        - If we are behind (equity < 40%), push toward fold (-0.05).
        No adjustments in the middle: we're committed anyway.

    High SPR (deep stacks):
        - Draws in the 30–50% equity range gain implied odds value (+0.04).
        - Monster hands lose a little EV compared to shallow because opponents
          can fold more often before committing all chips (-0.03).

    Args:
        spr:    Stack-to-Pot Ratio (from compute_spr).
        equity: Raw Monte Carlo or table equity for this hand.

    Returns:
        Float adjustment to add to equity (can be negative).
    """
    adj = 0.0

    if spr < SPR_LOW:
        if equity >= 0.56:
            adj += 0.04    # Already ahead; push toward commitment
        elif equity < 0.42:
            adj -= 0.04    # Behind; discourage calling off chips

    elif spr > SPR_HIGH:
        if 0.30 <= equity < 0.54:
            adj += 0.04    # Drawing hands gain implied-odds value
        elif equity > 0.66:
            adj -= 0.03    # Opponents can fold before getting it all in vs. monsters

    else:
        if 0.45 <= equity < 0.56:
            adj += 0.02    # Moderate equity hands gain a small boost in mid SPR

    return adj


# =============================================================================
# SECTION 9: OPPONENT MODEL
#
# The OpponentModel class tracks statistics about the opponent across the
# entire session. It is persistent — it is created once in Player.__init__()
# and updated after every hand.
#
# Key statistics tracked:
#   - Pre-flop aggression rate (VPIP: Voluntarily Put In Pot)
#   - Post-flop bet frequency per street (flop/turn/river)
#   - Fold rate (did they fold before showdown?)
#   - Auction results (who won, did winner bet immediately after?)
#
# The model classifies the opponent into archetypes:
#   Tight/Loose (range width) × Passive/Aggressive (betting frequency)
# These archetypes feed into AdaptiveStrategy to adjust thresholds.
# =============================================================================

class OpponentModel:
    """
    Persistent tracker for opponent behavior statistics across a full session.

    All "record_*" methods are called from Player.on_hand_end() after each hand
    completes. Property accessors are called by AdaptiveStrategy to compute
    adjusted thresholds.
    """

    def __init__(self):
        # ---- Long-term global counters (never reset, session-wide) ----

        self.hands_seen     = 0   # Total hands played this session
        self.pf_total       = 0   # Hands where we observed pre-flop action

        # Per-street tracking: {street: count of times we observed the street}
        self.postflop_opps = {'flop': 0, 'turn': 0, 'river': 0}

        self.showdowns   = 0   # Hands that reached the river

        # Auction-specific counters.
        self.auction_rounds    = 0   # Total auction phases seen
        self.our_bids          = deque(maxlen=200)  # Rolling deque of bid values

        # ---- Post-auction behavior tracking ----
        # Tracks how often the opponent bets the flop immediately after winning
        # the auction — a signal of whether they use the revealed card actively.
        self.opp_won_auction_count      = 0   # Times opp won the auction

        # ---- Exponential Moving Average (EMA) State ----
        # Alpha dictates the decay rate. Equivalent to a ~100 hand window.
        self.ALPHA = 0.04   # ~4% weight to the most recent hand, ~96% to the existing average
        
        # Initialize EMA trackers as None so we can seed them with the first observation
        self.ema_vpip = None
        self.ema_fold_rate = None
        self.ema_opp_auct_bet_rate = None
        self.ema_postflop_agg = None
        self.ema_won_auction = None

    def record_preflop(self, was_aggressive: bool):
        """
        Log whether the opponent played aggressively pre-flop (raised, 3-bet,
        or made any bet beyond completing the blind).

        Args:
            was_aggressive: True if opponent voluntarily put in chips pre-flop.
        """
        self.pf_total     += 1
        
        # Update EMA
        val = float(was_aggressive)
        if self.ema_vpip is None:
            self.ema_vpip = val
        else:
            self.ema_vpip = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_vpip

    def record_street_bet(self, street: str, opp_bet: bool):
        """
        Record whether the opponent made a bet or raise on a given post-flop street.

        Called from Player.on_hand_end() for each street that was actually played.

        Args:
            street:  One of 'flop', 'turn', 'river'.
            opp_bet: True if the opponent bet/raised on that street.
        """
        if street not in self.postflop_opps:
            return   # Ignore unknown streets (safety guard)

        self.postflop_opps[street] += 1

        # Update EMA
        val = float(opp_bet)
        if self.ema_postflop_agg is None:
            self.ema_postflop_agg = val
        else:
            self.ema_postflop_agg = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_postflop_agg

    def record_hand_end(self, payoff: int, final_street: str, opp_revealed: list):
        """
        Update session-wide counters at the conclusion of each hand.

        Determines whether the opponent folded (we won before showdown),
        whether the hand reached showdown, and whether the opponent's cards
        were revealed (meaning we won the auction this hand).

        Args:
            payoff:       Our chip gain/loss for this hand.
            final_street: The last street played ('pre-flop', 'flop', ..., 'river').
            opp_revealed: List of cards the opponent showed (empty if none).
        """
        self.hands_seen += 1

        # If we won chips and the hand didn't reach the river, the opponent folded.
        won_early = payoff > 0 and final_street != 'river'

        if final_street == 'river':
            self.showdowns += 1

        # If the opponent's card was revealed, we must have won the auction.
        auc = float(opp_revealed != [])  # 1.0 if we won the auction, 0.0 if not
        if self.ema_won_auction is None:
            self.ema_won_auction = auc
        else:
            self.ema_won_auction = self.ALPHA * auc + (1.0 - self.ALPHA) * self.ema_won_auction

        # Update EMA
        val = float(won_early)
        if self.ema_fold_rate is None:
            self.ema_fold_rate = val
        else:
            self.ema_fold_rate = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_fold_rate

    def record_auction(self, our_bid: int):
        """
        Record the bid our bot placed in the auction phase.

        Bids are stored in a capped list to allow analysis of bidding patterns
        (though this is currently informational only — bid_multiplier() uses
        win rate, not bid amounts).

        Args:
            our_bid: The chip amount we bid.
        """
        self.auction_rounds += 1
        self.our_bids.append(our_bid)

        # deque(maxlen=200) automatically discards the oldest entry.

    def record_opp_auction_result(self, opp_won: bool, opp_bet_after: bool):
        """
        Record whether the opponent won the auction and, if so, whether they
        immediately bet on the subsequent flop.

        Tracking post-auction bet behavior lets us infer whether the opponent
        uses the revealed card information aggressively (bluffing/value betting)
        or passively (trapping/slow-playing).

        Args:
            opp_won:      True if the opponent won the auction (they saw a card).
            opp_bet_after: True if the opponent bet on the flop after winning.
        """
        if opp_won:
            self.opp_won_auction_count += 1

            # Update EMA
            val = float(opp_bet_after)
            if self.ema_opp_auct_bet_rate is None:
                self.ema_opp_auct_bet_rate = val
            else:
                self.ema_opp_auct_bet_rate = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_opp_auct_bet_rate

    # ---- Computed rate properties ----

    @property
    def our_auction_win_rate(self) -> float:
        """
        Fraction of auction phases where our bot won and received a card reveal.
        Starts at 0.4 (bet more aggressively) before any auctions are seen.
        """
        return self.ema_won_auction if self.auction_rounds > 8 else 0.4

    @property
    def opp_post_auction_bet_rate(self) -> float:
        """
        How often the opponent bets the flop immediately after winning an auction.

        Uses the recent ema if at least 16 data points exist;
        otherwise defaults to 0.6 as a neutral assumption (slightly aggressive) until we have enough data.

        High rate → opponent uses auction information aggressively.
        Low rate  → opponent may be slow-playing or the auction card didn't help.
        """
        if self.opp_won_auction_count < 16 or self.ema_opp_auct_bet_rate is None:
            return 0.6 # Default to slightly aggressive until we have data
        return self.ema_opp_auct_bet_rate

    @property
    def recent_vpip(self) -> float:
        """
        Recent pre-flop looseness using the exponential moving average.
        Defaults to 0.6 (loose) until we have enough data to trust the EMA.
        """
        if self.pf_total < 16 or self.ema_vpip is None:
            return 0.6  # Default to loose until we have data
        return self.ema_vpip

    @property
    def recent_fold_rate(self) -> float:
        """
        Recent fold rate using only the last _WINDOW hands.
        Falls back to session-wide fold_rate if insufficient data.
        """
        if self.hands_seen < 16 or self.ema_fold_rate is None:
            return 0.16 # Default to a moderate fold rate until we have data
        return self.ema_fold_rate
    
    @property
    def postflop_aggression(self) -> float:
        """
        Overall post-flop aggression using exponential decay.
        High = aggressive, Low = passive/checking.
        """
        to = sum(self.postflop_opps.values())
        
        # Fall back to the raw session average if we have less than 16 post-flop observations
        if to < 16 or self.ema_postflop_agg is None:
            return 0.4 # Default to a moderate aggression rate until we have data
        return self.ema_postflop_agg

    # ---- Archetype classifiers ----

    def is_tight(self)      -> bool: return self.recent_vpip < 0.30           # Plays <30% of hands
    def is_loose(self)      -> bool: return self.recent_vpip > 0.70           # Plays >70% of hands
    def is_aggressive(self) -> bool: return self.postflop_aggression > 0.50   # Bets >50% of streets
    def is_passive(self)    -> bool: return self.postflop_aggression < 0.30   # Bets <30% of streets
    def is_folder(self)     -> bool: return self.recent_fold_rate > 0.30      # Folds >35% pre-showdown
    def sufficient_data(self) -> bool: return self.hands_seen >= MODEL_MIN_HANDS  # Trust the model?

    def inferred_range_fraction(self) -> float:
        """
        Translate the opponent's observed play style into a range fraction
        used to constrain Monte Carlo opponent hand sampling.

        Returns:
            RANGE_TIGHT  (0.30) if opponent plays few hands (tight).
            RANGE_LOOSE  (0.96) if opponent plays many hands (loose) OR
                                 if we don't yet have enough data to trust the model.
            RANGE_MEDIUM (0.50) for everything in between.
        """
        if not self.sufficient_data():
            return RANGE_LOOSE   # Default to no constraint until we have data

        if self.is_tight():
            return RANGE_TIGHT
        if self.is_loose():
            return RANGE_LOOSE
        return RANGE_MEDIUM

    def __str__(self):
        """Debug-friendly string showing key model statistics."""
        return (f'Hands={self.hands_seen}  VPIP={self.ema_vpip:.2f}  '
                f'PF_agg={self.postflop_aggression:.2f}  '
                f'Fold%={self.fold_rate:.2f}  '
                f'AuctWin%={self.our_auction_win_rate:.2f}  '
                f'OppPostAuctBet%={self.opp_post_auction_bet_rate:.2f}')


# =============================================================================
# SECTION 10: ADAPTIVE STRATEGY ENGINE
#
# AdaptiveStrategy sits on top of the OpponentModel and converts behavioral
# statistics into concrete threshold values that govern our actions.
#
# The five thresholds it produces:
#   fold_t:  Fold when equity is below this (facing a bet)
#   call_t:  Call when equity is at least this (facing a bet)
#   value_t: Value bet/raise when equity is at least this
#   bluff_f: Probability of bluffing with a weak hand (random check)
#   small_f: Fraction of pot for thin/merged bets
#   big_f:   Fraction of pot for strong value bets
#
# These are initialized to global constants and then adjusted by:
#   1. Opponent archetype (tight/loose/passive/aggressive)
#   2. Table position (Big Blind / Small Blind)
#   3. Post-auction signals (did they use their auction advantage?)
# =============================================================================

class AdaptiveStrategy:
    """
    Dynamic strategy layer that adjusts decision thresholds based on opponent
    tendencies observed by OpponentModel.
    """

    def __init__(self, model: OpponentModel):
        """
        Args:
            model: The shared OpponentModel instance (same object across the session).
        """
        self.model = model

    def thresholds(self, is_bb: bool = False,
                   opp_won_auction_and_bet: bool = False) -> tuple:
        """
        Compute the six threshold values for the current action decision.

        Process:
            1. Start with global baseline constants.
            2. If insufficient opponent data → apply only positional adjustments.
            3. Otherwise → apply opponent-archetype adjustments first.
            4. Then apply positional modifiers.
            5. Then apply post-auction-aggression modifiers if relevant.

        Args:
            is_bb:                  True if we are the Big Blind this hand.
            opp_won_auction_and_bet: True if opponent won auction AND bet flop.

        Returns:
            Tuple: (fold_t, call_t, value_t, bluff_f, small_f, big_f)
        """
        m = self.model

        # Initialize all thresholds from global constants.
        fold_t  = FOLD_THRESHOLD
        call_t  = CALL_THRESHOLD
        value_t = VALUE_THRESHOLD
        bluff_f = BLUFF_FREQ
        small_f = BET_SMALL_FRAC
        big_f   = BET_BIG_FRAC

        # --- Insufficient data: only apply positional adjustments ---
        if not m.sufficient_data():
            fold_t, call_t, bluff_f = self._apply_position(fold_t, call_t, bluff_f, is_bb)
            if opp_won_auction_and_bet:
                fold_t, call_t = self._apply_opp_auction_aggression(fold_t, call_t, m)
            return fold_t, call_t, value_t, bluff_f, small_f, big_f

        # --- Continuous archetype blending ---
        # Compute continuous tightness/aggression axes in [0, 1] using sigmoid
        # smoothing. This replaces hard binary cutoffs with smooth transitions.
        #   tightness: 0 = tight, 1 = loose
        #   aggression: 0 = passive, 1 = aggressive
        vpip = m.recent_vpip
        agg  = m.postflop_aggression

        def _sigmoid(x, mid, steepness=12.0):
            return 1.0 / (1.0 + math.exp(-steepness * (x - mid)))

        tightness  = _sigmoid(vpip, 0.5)   # midpoint between tight<0.30 and loose>0.70
        aggression = _sigmoid(agg,  0.4)    # midpoint between passive<0.30 and aggro>0.50

        # Bilinear blend weights for the four archetypes.
        w_tp = (1.0 - tightness) * (1.0 - aggression)   # tight-passive
        w_ta = (1.0 - tightness) * aggression            # tight-aggressive
        w_lp = tightness * (1.0 - aggression)            # loose-passive
        w_la = tightness * aggression                     # loose-aggressive

        # Target threshold values for each archetype (match original discrete values).
        #    (     fold_t,                call_t,                    value_t,         bluff_f,          small_f,        big_f)
        tp = (FOLD_THRESHOLD,        CALL_THRESHOLD,        VALUE_THRESHOLD + 0.04, BLUFF_FREQ * 1.60, 0.40,           0.60)
        ta = (FOLD_THRESHOLD - 0.05, CALL_THRESHOLD - 0.04, VALUE_THRESHOLD - 0.05, BLUFF_FREQ - 0.06, BET_SMALL_FRAC, BET_BIG_FRAC)
        lp = (FOLD_THRESHOLD,        CALL_THRESHOLD - 0.04, VALUE_THRESHOLD - 0.10, BLUFF_FREQ / 4,    0.65,           1.00)
        la = (FOLD_THRESHOLD + 0.05, CALL_THRESHOLD + 0.06, VALUE_THRESHOLD + 0.04, BLUFF_FREQ / 3,    BET_SMALL_FRAC, BET_BIG_FRAC)

        # Weighted blend of all six thresholds.
        fold_t  = w_tp * tp[0] + w_ta * ta[0] + w_lp * lp[0] + w_la * la[0]
        call_t  = w_tp * tp[1] + w_ta * ta[1] + w_lp * lp[1] + w_la * la[1]
        value_t = w_tp * tp[2] + w_ta * ta[2] + w_lp * lp[2] + w_la * la[2]
        bluff_f = w_tp * tp[3] + w_ta * ta[3] + w_lp * lp[3] + w_la * la[3]
        small_f = w_tp * tp[4] + w_ta * ta[4] + w_lp * lp[4] + w_la * la[4]
        big_f   = w_tp * tp[5] + w_ta * ta[5] + w_lp * lp[5] + w_la * la[5]

        # Folder bonus: additive on top of blended result (same condition as before,
        # but using continuous weights to avoid triggering vs. LAG).
        if m.is_folder() and not (tightness > 0.5 and aggression > 0.5):
            bluff_f = bluff_f * 1.3  # Increase bluff frequency by 30% against opponents who fold a lot pre-showdown

        # --- Apply positional modifiers ---
        fold_t, call_t, bluff_f = self._apply_position(fold_t, call_t, bluff_f, is_bb)

        # --- Apply post-auction aggression modifiers if relevant ---
        if opp_won_auction_and_bet:
            fold_t, call_t = self._apply_opp_auction_aggression(fold_t, call_t, m)

        return fold_t, call_t, value_t, bluff_f, small_f, big_f

    @staticmethod
    def _apply_position(fold_t: float, call_t: float,
                        bluff_f: float, is_bb: bool) -> tuple:
        """
        Adjust thresholds based on positional advantage or disadvantage.

        Big Blind (Out of Position post-flop):
            - Must fold more often when facing bets (no information advantage).
            - Tighten the call requirement.
            - Reduce bluff frequency (bluffs fail more OOP).

        Small Blind (In Position post-flop):
            - Can float wider and bluff with more success.
            - Slightly lower fold threshold.
            - Slightly increase bluff frequency.

        Args:
            fold_t, call_t, bluff_f: Current threshold values.
            is_bb: True if we are the Big Blind.

        Returns:
            Adjusted (fold_t, call_t, bluff_f) tuple.
        """
        if is_bb:
            # Out of position: tighten ranges
            fold_t  = min(fold_t  + 0.03, 0.40)
            call_t  = min(call_t  + 0.02, 0.55)
            bluff_f = max(bluff_f * POSITION_OOP_BLUFF_SCALE, 0.05)
        else:
            # In position: loosen up slightly, bluff more
            fold_t  = max(fold_t  - 0.02, 0.18)
            bluff_f = min(bluff_f * POSITION_IP_BLUFF_SCALE, 0.45)

        return fold_t, call_t, bluff_f

    def _apply_opp_auction_aggression(self, fold_t: float,
                                       call_t: float,
                                       m: 'OpponentModel') -> tuple:
        """
        Adjust fold/call thresholds based on how often the opponent bets after
        winning the auction. Called only when opp_won_auction_and_bet is True.

        If the opponent frequently bets after winning the auction:
            → They are likely betting with actual strength → tighten our ranges.
        If they rarely bet after winning:
            → They may be trapping or bluffing infrequently → allow lighter calls.
        If we don't have enough data yet (< 5 observations):
            → Apply a conservative, diluted tightening as a precaution.

        Args:
            fold_t, call_t: Current threshold values.
            m: The OpponentModel for statistics.

        Returns:
            Adjusted (fold_t, call_t) tuple.
        """
        # Conservative default: not enough data to characterize behavior.
        if m.opp_won_auction_count < 5:
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP * 0.5, 0.45)
            return fold_t, call_t

        rate = m.opp_post_auction_bet_rate

        if rate >= OPP_AUCT_AGG_HIGH:
            # Opponent almost always bets after winning → their hand is real.
            # Raise both fold and call thresholds to play tighter.
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP, 0.48)
            call_t = min(call_t + 0.04, 0.58)

        elif rate <= OPP_AUCT_AGG_LOW:
            # Opponent rarely bets after winning → they may be slow-playing
            # or the card didn't help much. Allow lighter calls.
            call_t = max(call_t - OPP_AUCT_CALL_DIP, 0.33)

        return fold_t, call_t

    def bid_multiplier(self) -> float:
        """
        Scale our raw computed auction bid based on our recent auction win rate.

        If we are losing too many auctions (win rate < 25%), bid higher to
        compete more aggressively and gain more information.

        If we are winning almost all auctions (win rate > 75%), bid lower
        to avoid overpaying for information we tend to get anyway.

        Returns:
            Float multiplier applied to the raw computed bid amount.
        """
        m = self.model

        # Not enough auction history to adjust bidding behavior.
        if m.auction_rounds < 15:
            return 1.0

        wr = m.our_auction_win_rate

        if wr < 0.30:   return 1.60   # Losing lots → bid 60% more aggressively
        elif wr < 0.40: return 1.25   # Slightly low win rate → bid 25% more
        elif wr > 0.8: return 0.70   # Winning most → back off, save chips
        elif wr > 0.6: return 0.90   # Winning plenty → slight pullback
        return 1.0                    # Balanced win rate → no adjustment

    def describe(self) -> str:
        """
        Return a human-readable description of the opponent's inferred archetype.
        Useful for logging and debugging.

        Returns:
            String like 'Tight-Aggressive' or 'Loose-Passive-(Folder)'.
        """
        m = self.model
        if not m.sufficient_data():
            return 'Insufficient data — using defaults'

        parts = []
        if m.is_tight():        parts.append('Tight')
        elif m.is_loose():      parts.append('Loose')
        else:                   parts.append('Normal')

        if m.is_passive():      parts.append('Passive')
        elif m.is_aggressive(): parts.append('Aggressive')
        else:                   parts.append('Balanced')

        if m.is_folder():       parts.append('(Folder)')

        return '-'.join(parts)


# =============================================================================
# SECTION 11: STREET-AWARE BET SIZING
#
# Different streets warrant different bet sizes. Early streets (flop) use
# smaller sizes to build pots gently; later streets (river) use larger sizes
# for maximum extraction. The sizing also shifts based on board wetness:
# bet smaller on wet boards (to avoid charging draws less than pot odds isn't
# the intent here — rather, we size down to avoid over-betting into bad equity).
# =============================================================================

# Base (thin, strong) bet size fractions per street.
# Thin sizing = used for marginal value hands or bluffs.
# Strong sizing = used for premium value hands.
_STREET_SIZING = {
    'pre-flop': (0.4, 0.6),   # Half-pot to three-quarter pot pre-flop
    'flop'    : (0.40, 0.55),   # Slightly under half-pot on the flop
    'turn'    : (0.55, 0.70),   # Growing as pot builds
    'river'   : (0.65, 0.90),   # Large bets to extract maximum value at showdown
}


def street_sizing(street: str, texture: BoardTexture, strong: bool) -> float:
    """
    Determine the appropriate bet sizing fraction for the current street and
    board texture.

    Logic:
        1. Look up (thin, big) fractions for the current street.
        2. Select thin or big based on whether we have a strong hand.
        3. Apply a wetness adjustment: wet boards → size down slightly;
           dry boards → size up slightly. This reflects that on wet boards,
           opponent draws make large bets more likely to get called by equity
           share, and we want to keep the pot manageable.

    Args:
        street:  Current street name.
        texture: Board texture object with wetness score.
        strong:  True if we have a value hand, False for thin/bluff sizing.

    Returns:
        Float fraction of the pot to bet (e.g., 0.65 = 65% pot bet).
    """
    thin_f, big_f = _STREET_SIZING.get(street, (BET_SMALL_FRAC, BET_BIG_FRAC))
    base = big_f if strong else thin_f

    # Wet board adjustment: scale bet size inversely with wetness.
    # wetness=0.5 → no change; wetness=1.0 → -0.10; wetness=0.0 → +0.10
    wet_adj = (texture.wetness - 0.5) * (-0.20)

    return max(0.30, min(1.20, base + wet_adj))


def is_drawing_hand(hole: list, board: list, texture: BoardTexture) -> bool:
    """
    Detect whether our hand is primarily a draw (flush draw or open-ended
    straight draw) rather than a made hand.

    Checks:
      1. Flush draw: at least 1 hole card shares a suit with enough board cards
         to make 4-to-a-flush total.
      2. Straight draw: at least 4 unique ranks within a 5-rank window, with at
         least 1 of our hole cards contributing.

    Args:
        hole:    Our 2 hole cards.
        board:   Community cards (3+).
        texture: Pre-computed BoardTexture.

    Returns:
        True if we appear to be on a significant draw.
    """
    if len(board) < 3:
        return False

    # --- Flush draw check ---
    hole_suits = [card_suit(c) for c in hole]
    board_suits = [card_suit(c) for c in board]
    for s in SUITS:
        hole_count = hole_suits.count(s)
        board_count = board_suits.count(s)
        if hole_count >= 1 and hole_count + board_count >= 4:
            return True

    # --- Straight draw check ---
    all_ranks = sorted(set(card_rank(c) for c in hole + board))
    hole_rank_set = set(card_rank(c) for c in hole)

    # Check every 5-rank window for 4+ ranks with at least 1 hole card.
    for base in range(2, 11):
        window = set(range(base, base + 5))
        in_window = [r for r in all_ranks if r in window]
        if len(in_window) >= 4 and any(r in window for r in hole_rank_set):
            return True

    # Wheel check (A-2-3-4-5): treat Ace as 1.
    wheel = {14, 2, 3, 4, 5}
    wheel_ranks = [r for r in all_ranks if r in wheel]
    if len(wheel_ranks) >= 4 and any(r in wheel for r in hole_rank_set):
        return True

    return False


# =============================================================================
# SECTION 12: AUCTION CARD EXPLOITATION
#
# When we win the auction, we learn one of the opponent's hole cards.
# This section defines how we adjust our behavior based on the rank of
# that revealed card. High card → opponent has partial strength; treat
# cautiously. Low card → opponent's hand is weaker; exploit accordingly.
# =============================================================================

def revealed_card_adjustment(revealed_card: str, board: list, auctions_won: int = 0) -> tuple:
    """
    Compute equity, bluff, and sizing adjustments based on the rank of the
    opponent's card that was revealed through winning the auction.

    Three scenarios:

    1. High card (Jack+, rank ≥ 11):
       The opponent holds at least one strong card. We reduce our equity
       estimate, bluff less (harder to fold them), and bet smaller (less
       value in betting into a potentially strong hand).

    2. Low card (7 or below, rank ≤ 7):
       The opponent holds a weak card. We gain equity, bluff more, and
       bet bigger. However, if that low card connects to the board (within
       2 ranks of a board card), we dampen the boost because even low cards
       can be strong in context (e.g., a paired board or straight draws).

    3. Middle card (8–10):
       Neutral — no significant adjustment either direction.

    Args:
        revealed_card: The card string shown to us from the auction win.
        board:         Current community cards.

    Returns:
        Tuple of (equity_delta, bluff_scale, size_scale) where:
            equity_delta: Added to effective equity (can be negative).
            bluff_scale:  Multiply bluff frequency by this.
            size_scale:   Multiply bet size fraction by this.
    """
    rank = card_rank(revealed_card)
    board_ranks = [card_rank(c) for c in board]

    # Check if the low card could connect to the board (straight/pair proximity).
    connects_to_board = any(abs(rank - br) <= 2 for br in board_ranks)

    # Confidence scalar: ramps from 0.0 → 1.0 as auctions_won approaches
    # AUCTION_CONFIDENCE_MIN. Below that threshold, adjustments are dampened.
    confidence = min(1.0, auctions_won / AUCTION_CONFIDENCE_MIN)

    if rank >= REVEALED_HIGH_RANK:
        # Opponent has a Jack, Queen, King, or Ace → tread carefully.
        eq_d  = REVEALED_HIGH_EQ_DELTA    * confidence
        bl_sc = 1.0 + (REVEALED_HIGH_BLUFF_SCALE - 1.0) * confidence
        sz_sc = 1.0 + (REVEALED_HIGH_SIZE_SCALE  - 1.0) * confidence
        return (eq_d, bl_sc, sz_sc)

    elif rank <= REVEALED_LOW_RANK:
        # Opponent has a 7 or below → they are weak, but dampen if it connects.
        dampen = 0.5 if connects_to_board else 1.0
        eq_d  = REVEALED_LOW_EQ_DELTA * dampen * confidence
        bl_sc = 1.0 + (REVEALED_LOW_BLUFF_SCALE - 1.0) * dampen * confidence
        sz_sc = 1.0 + (REVEALED_LOW_SIZE_SCALE  - 1.0) * dampen * confidence
        return (eq_d, bl_sc, sz_sc)

    # Middle rank (8–10): no meaningful signal, return neutral adjustments.
    return (0.0, 1.0, 1.0)


# =============================================================================
# SECTION 13: AUCTION BIDDING ENGINE
#
# The auction mechanic allows players to bid for the right to see one of the
# opponent's hole cards. The winner pays their bid and the information is
# revealed. Our bidding strategy is grounded in Expected Value:
#
#   EV of winning auction   = equity if we see a card (better decisions post-flop)
#   EV of losing auction    = equity if opponent sees their own card (they play better)
#   Information value       = |eq_win - eq_lose| × pot_size
#   Final bid               = information_value × marginal_factor × adaptive_multiplier
#
# Additional constraints:
#   - Bid SMALL_BLIND if our equity is at the extreme ends (>85% or <15%) — information
#     doesn't change our decisions much at these extremes.
#   - Bid more as BB (out of position) because information is more valuable
#     when you act first and have less information.
#   - Cap the bid as a fraction of the pot to avoid giving away too many chips.
# =============================================================================

def dynamic_max_bid_frac(pot: int, effective_stack: int) -> float:
    """
    Compute the maximum fraction of the pot we are willing to bid.

    The cap shrinks as the pot grows (we've already committed chips; the
    marginal value of information decreases). The cap grows slightly if
    stacks are very deep (more implied value from better decisions post-flop).

    The result is clamped to [BID_CAP_MIN, BID_CAP_MAX].

    Args:
        pot:             Current pot size in chips.
        effective_stack: Smaller of the two players' remaining stacks.

    Returns:
        Float fraction of the pot representing the maximum bid.
    """
    pot_fraction = pot / max(STARTING_STACK, 1)

    # As pot grows to 30%+ of starting stack, pot_cap approaches BID_CAP_MIN.
    pot_cap = BID_CAP_MAX - (BID_CAP_MAX - BID_CAP_MIN) * min(pot_fraction / 0.30, 1.0)

    # Deep stack bonus: if effective stack is more than 30% of starting stack,
    # allow a slightly higher bid cap (more post-flop streets to exploit info).
    stack_fraction = effective_stack / max(STARTING_STACK, 1)
    stack_adj = 0.04 * max(0.0, stack_fraction - 0.3)

    return min(BID_CAP_MAX, max(BID_CAP_MIN, pot_cap + stack_adj))


# Tuning constants for the auction equity estimation sub-routines.
AUC_BATCH_TRIALS = 50  # Total MC trials for batched auction equity estimation
AUC_LOSE_FRAC    = 0.35 # Fallback range tightening if no opponent data (35% of range)


def _dynamic_lose_frac(opp_range_fraction: float, strategy: AdaptiveStrategy) -> float:
    """
    Determine how much the opponent tightens their range when they win the auction.

    Formula:
        dynamic_multiplier = 0.25 + 0.42 * agg_rate
        - agg_rate = 0.0 → multiplier = 0.25 → keep 25% of range (very tight)
        - agg_rate = 1.0 → multiplier = 0.67 → keep 67% of range (fairly wide)

    Falls back to 50% of original range fraction before sufficient data exists.

    Args:
        opp_range_fraction: Inferred range width before auction.
        strategy:           AdaptiveStrategy (provides access to OpponentModel).

    Returns:
        Float — estimated range fraction after opponent wins auction.
    """
    model = strategy.model

    base_frac = opp_range_fraction * 0.50

    if model.sufficient_data():
        agg_rate           = model.opp_post_auction_bet_rate
        dynamic_multiplier = 0.25 + (0.42 * agg_rate)
        base_frac          = opp_range_fraction * dynamic_multiplier

    return max(0.10, min(opp_range_fraction, base_frac))


def _batched_auction_equity(hole: list, board: list,
                            excluded: set,
                            opp_range_fraction: float,
                            strategy: AdaptiveStrategy) -> tuple:
    """
    Estimate equity for both auction outcomes (win / lose) in a single
    batched Monte Carlo pass, eliminating redundant setup work.

    For each trial:
      - Sample an opponent hand from their range.
      - Win scenario: reveal one card, run the board out, compare hands.
      - Lose scenario: opponent plays a tighter range, run the board out.

    Args:
        hole:               Our hole cards.
        board:              Current community cards (usually empty pre-flop).
        excluded:           Set of dead cards.
        opp_range_fraction: Opponent's baseline range width.
        strategy:           AdaptiveStrategy (for lose-scenario range tightening).

    Returns:
        Tuple (eq_win, eq_lose): estimated equity if we win / lose the auction.
    """
    remaining = [c for c in FULL_DECK if c not in excluded]
    range_pairs = build_opponent_range(opp_range_fraction, excluded)

    if not range_pairs:
        range_pairs = [(a, b) for a in remaining for b in remaining if a < b]

    # Tighter range for lose scenario.
    tighter_frac = _dynamic_lose_frac(opp_range_fraction, strategy)
    tighter_pairs = build_opponent_range(tighter_frac, excluded)
    if not tighter_pairs:
        tighter_pairs = range_pairs

    cards_needed_board = 5 - len(board)

    # Pre-convert known cards.
    e7_hole = [_e7(c) for c in hole]
    e7_board = [_e7(c) for c in board]
    _eval = eval7.evaluate

    wins_win  = 0.0
    wins_lose = 0.0
    valid_win  = 0
    valid_lose = 0

    _choice = random.choice
    _sample = random.sample

    for _ in range(AUC_BATCH_TRIALS):
        # --- Win scenario: sample opp hand, reveal one card ---
        pair_w = _choice(range_pairs)
        opp_hand_w = list(pair_w)

        pool_w = [c for c in remaining if c not in opp_hand_w]
        if len(pool_w) >= cards_needed_board:
            board_completion_w = _sample(pool_w, cards_needed_board)
            e7_bc_w = [_e7(c) for c in board_completion_w]
            e7_opp_w = [_e7(c) for c in opp_hand_w]

            my_w  = _eval(e7_hole + e7_board + e7_bc_w)
            opp_w = _eval(e7_opp_w + e7_board + e7_bc_w)
            if my_w > opp_w:
                wins_win += 1.0
            elif my_w == opp_w:
                wins_win += 0.5
            valid_win += 1

        # --- Lose scenario: opponent plays tighter range ---
        pair_l = _choice(tighter_pairs)
        opp_hand_l = list(pair_l)

        pool_l = [c for c in remaining if c not in opp_hand_l]
        if len(pool_l) >= cards_needed_board:
            board_completion_l = _sample(pool_l, cards_needed_board)
            e7_bc_l = [_e7(c) for c in board_completion_l]
            e7_opp_l = [_e7(c) for c in opp_hand_l]

            my_l  = _eval(e7_hole + e7_board + e7_bc_l)
            opp_l = _eval(e7_opp_l + e7_board + e7_bc_l)
            if my_l > opp_l:
                wins_lose += 1.0
            elif my_l == opp_l:
                wins_lose += 0.5
            valid_lose += 1

    eq_win  = wins_win  / max(valid_win,  1)
    eq_lose = wins_lose / max(valid_lose, 1)
    return (eq_win, eq_lose)


def compute_bid(state: PokerState,
                strategy: AdaptiveStrategy,
                opp_range_fraction: float = RANGE_LOOSE) -> int:
    """
    Compute the optimal auction bid based on the Expected Value of information.

    Algorithm:
        1. Compute baseline equity without any information advantage.
        2. Short-circuit to bid 0 if equity is extreme (≥84% or ≤16%):
           at these extremes, knowing one card rarely changes our best action.
        3. Estimate eq_win = equity if we see the opponent's card.
        4. Estimate eq_lose = equity if opponent sees their own card.
        5. Information value = |eq_win - eq_lose| × pot (in chips).
        6. Apply marginal_factor = 4 × eq × (1-eq): peaks at 50% equity, with minimum 0.64 cap
           because near-even equities benefit most from information.
        7. Apply adaptive multiplier from AdaptiveStrategy (based on auction
           win rate history).
        8. Multiply by OOP_BID_MULTIPLIER if we are the Big Blind.
        9. Enforce minimum bid and clamp to dynamic pot-fraction cap.

    Args:
        state:              Current game state (provides pot, chips, etc.).
        strategy:           AdaptiveStrategy for bid multiplier lookup.
        opp_range_fraction: Opponent range constraint.

    Returns:
        Integer chip amount to bid (may be 0 to skip the auction).
    """
    hole  = list(state.my_hand)
    board = list(state.board)              if state.board              else []
    opp   = list(state.opp_revealed_cards) if state.opp_revealed_cards else []

    # Step 1: Baseline equity (no auction information).
    equity = monte_carlo_equity(hole, board, opp,
                                n_sims=MC_SIMS_AUCTION,
                                opp_range_fraction=opp_range_fraction)

    # Step 2: Information is irrelevant at the extremes — we know what to do.
    if equity >= 0.84 or equity <= 0.16:
        return SMALL_BLIND

    excluded = set(hole + board + opp)

    # Steps 3 & 4: Estimate equity in both auction outcomes (batched).
    eq_win, eq_lose = _batched_auction_equity(hole, board, excluded,
                                              opp_range_fraction, strategy)

    # Step 5: Raw information value in chips.
    info_value = abs(eq_win - eq_lose) * state.pot

    # Step 6: Marginal factor — peaks at equity=0.50 (maximum uncertainty).
    # 4 × 0.5 × 0.5 = 1.0 (maximum); Cap at 0.64 to prevent overbidding on very strong marginal hands.
    marginal_factor = min(4.0 * equity * (1.0 - equity), 0.64)

    # Step 7: Adaptive bid multiplier from auction win rate history.
    adapt_mult = strategy.bid_multiplier()
    raw_bid    = info_value * marginal_factor * adapt_mult

    # Step 8: Out-of-position premium (BB bids more for information).
    if state.is_bb:
        raw_bid *= OOP_BID_MULTIPLIER

    # Enforce a minimum bid: at least 1 big blind if our auction win rate is
    # poor (< 30%), otherwise 1 small blind. Prevents bidding trivially small amounts.
    min_bid  = BIG_BLIND
    if strategy.model.our_auction_win_rate <= 0.3:
        min_bid += SMALL_BLIND
    elif strategy.model.our_auction_win_rate > 0.7:
        min_bid -= SMALL_BLIND
    
    raw_bid  = max(raw_bid, min_bid)

    # Step 9: Cap the bid as a dynamic fraction of the pot.
    eff_stack    = min(state.my_chips, getattr(state, 'opp_chips', state.my_chips))
    dyn_cap_frac = dynamic_max_bid_frac(state.pot, eff_stack)
    max_bid      = int(state.pot * dyn_cap_frac)
    bid          = int(raw_bid)

    # Final clamp: bid must not exceed our chip stack or the pot cap.
    return min(bid, max_bid, state.my_chips)


# =============================================================================
# SECTION 14: CORE ACTION DECISION ENGINE
#
# decide_action() is the final output stage. It takes equity, adjusted
# thresholds, board texture, and auction context and returns a concrete
# game action: Fold, Check, Call, or Raise.
#
# Decision flow:
#   1. Unpack auction context flags.
#   2. Fetch adaptive thresholds from AdaptiveStrategy.
#   3. Apply auction-loss tightening if opponent won the auction.
#   4. Compute pot odds for calling decisions.
#   5. Adjust equity based on SPR, position, and revealed card.
#   6. Compute effective bluff frequency (texture-weighted).
#   7. If SPR is critically low → jam/fold immediately (skip nuanced logic).
#   8. Route 1 (we can check / we are the aggressor):
#       a. Strong hand → value raise.
#       b. Marginal hand → thin value raise.
#       c. Weak hand → stochastic bluff or check.
#   9. Route 2 (facing a bet from opponent):
#       a. Strong hand → reraise, or call if no raise available.
#       b. Marginal hand (meets pot odds) → call.
#       c. Weak hand → fold.
#  10. Fallback handlers ensure we never return None.
# =============================================================================

def estimate_shove_range(opp_model) -> float:
    """
    Estimate the range fraction an opponent shoves with pre-flop.
    Tighter than their general VPIP — shoving ranges are narrower.
    Falls back to a conservative default before sufficient data.
    """
    if not opp_model.sufficient_data():
        return 0.16   # Default: assume top 15% when no data

    # Shove range is roughly half their standard inferred range,
    # scaled down further if they are passive pre-flop.
    base_frac = opp_model.inferred_range_fraction() * 0.50
    if opp_model.preflop_aggression_rate < 0.30:
        base_frac *= 0.70   # Passive opponents shove even tighter

    return max(0.08, min(0.32, base_frac))

def decide_action(state: PokerState, equity: float,
                  strategy: AdaptiveStrategy,
                  texture: BoardTexture,
                  auction_ctx: dict | None = None):
    """
    Determine the best action to take given the current game state and context.

    This is the core strategic decision function. It synthesizes all analysis
    (equity, thresholds, SPR, texture, position, auction context) into a single
    concrete game action.

    Args:
        state:       Current game state object.
        equity:      Raw equity estimate from get_equity().
        strategy:    AdaptiveStrategy instance providing adjusted thresholds.
        texture:     BoardTexture object for the current board.
        auction_ctx: Optional dict containing auction-related flags:
                       'we_won_auction':       True if we saw the opponent's card.
                       'revealed_card':        The card string we saw (or None).
                       'opp_won_auction':      True if opponent saw their card.
                       'opp_bet_post_auction': True if opp bet flop after winning.
                       'is_bb':                True if we are the Big Blind.

    Returns:
        One of ActionFold, ActionCall, ActionCheck, ActionRaise.
    """
    # --- Step 1: Unpack auction context ---
    ctx = auction_ctx or {}
    we_won_auction       = ctx.get('we_won_auction', False)
    revealed_card        = ctx.get('revealed_card', None)
    opp_won_auction      = ctx.get('opp_won_auction', False)
    opp_bet_post_auction = ctx.get('opp_bet_post_auction', False)
    is_bb                = ctx.get('is_bb', False)
    street_reraise_count = ctx.get('street_reraise_count', 0)

    # Combined flag: opponent won the auction AND immediately bet → likely strong.
    opp_won_and_bet = opp_won_auction and opp_bet_post_auction

    # --- Step 2: Fetch adaptive thresholds ---
    fold_t, call_t, value_t, bluff_f, _sf, _bf = strategy.thresholds(
        is_bb=is_bb, opp_won_auction_and_bet=opp_won_and_bet)
    
    # Reraise pressure adjustment: each opponent reraise signals a stronger hand,
    # so require more equity before betting for value.
    if street_reraise_count > 0:
        value_t = min(0.90, value_t * (1.0 + 0.1 * street_reraise_count))

    # --- Step 3: Tighten ranges if opponent won the auction ---
    # When the opponent has superior information, we should be more conservative.
    # AUCTION_LOSS_K = 0.84 applies a 16% tightening effect.
    if opp_won_auction:
        # Direct multiplication works for bluff_f since it's a probability.
        bluff_f *= AUCTION_LOSS_K

        # Shift equity thresholds upward. Formula: new_t = 1 - K*(1-old_t)
        # This preserves the proportional distance from 1.0, just compressed.
        call_t  = 1.0 - AUCTION_LOSS_K * (1.0 - call_t)
        value_t = 1.0 - AUCTION_LOSS_K * (1.0 - value_t)

        # Ensure fold threshold doesn't accidentally exceed the new call threshold.
        fold_t = min(fold_t, call_t - 0.05)

    # --- Step 4: Basic pot geometry ---
    pot = state.pot
    ctc = state.cost_to_call   # Chips To Call (0 if no bet to face)
    hole = list(state.my_hand)

    # Pot odds: the minimum equity needed to break even on a call.
    pot_odds = ctc / (pot + ctc) if (ctc > 0 and pot + ctc > 0) else 0.0

    min_raise, max_raise = state.raise_bounds

    def clamp(amount: int, jitter: float = 0.04) -> int:
        """
        Clamp a proposed raise amount to the legal raise bounds.
        The engine requires raises to fall within [min_raise, max_raise].
        """
        noise = 1.0 + random.uniform(-jitter, jitter)
        jittered = int(amount * noise)
        return max(min_raise, min(jittered, max_raise))


    # --- Step 5: Effective equity (equity + all adjustments) ---

    # SPR adjustment: committed pots favor made hands; deep stacks favor draws.
    spr = compute_spr(state)
    adj = spr_adjustment(spr, equity)
    eff_eq = max(0.0, min(1.0, equity + adj))

    # Positional adjustment: being in position (SB) gives a small equity bonus;
    # being out of position (BB) gives a penalty.
    street = state.street
    if street != 'pre-flop':
        if is_bb:
            eff_eq = max(0.0, eff_eq + POSITION_OOP_EQ_DELTA)
        else:
            eff_eq = min(1.0, eff_eq + POSITION_IP_EQ_BONUS)
    elif street == 'pre-flop' and is_bb:
        # BB is last to act pre-flop → slight calling range bonus.
        call_t = max(call_t - POSITION_BB_CALL_BONUS, 0.30)
        value_t = value_t * 1.10  # Require stronger hands to 3-bet from the BB pre-flop

    # Revealed card adjustment: modify equity, bluff scale, and size scale
    # based on the specific card revealed from the auction win.
    rev_eq_delta = 0.0
    rev_bluff_sc = 1.0
    rev_size_sc  = 1.0
    if we_won_auction and revealed_card and street != 'pre-flop':
        board = list(state.board) if state.board else []
        rev_eq_delta, rev_bluff_sc, rev_size_sc = revealed_card_adjustment(
            revealed_card, board, auctions_won=strategy.model.ema_won_auction)
        eff_eq = max(0.0, min(1.0, eff_eq + rev_eq_delta))

    # --- Step 5b: Draw detection for sizing/threshold adjustments ---
    board_list = list(state.board) if state.board else []
    is_draw = False
    if street in ('flop', 'turn') and len(board_list) >= 3:
        is_draw = is_drawing_hand(list(state.my_hand), board_list, texture)

    # Draws get implied-odds bonus on flop/turn (lower call threshold).
    if is_draw:
        call_t = max(call_t - 0.05, 0.30)
        # Deep stacks amplify implied odds for draws.
        if spr > SPR_HIGH:
            call_t = max(call_t - 0.03, 0.27)

    # --- Step 6: Effective bluff frequency ---
    # Adjust bluffing frequency for board wetness:
    # Wet boards → bluff less (opponent has many draws and calls more).
    # Dry boards → bluff more (opponent has fewer continuing hands).
    # texture_bluff_scale: at wetness=0.5 → 1.0; at 0 → 1.2; at 1 → 0.8.
    texture_bluff_scale = 1.0 + 0.4 * (0.5 - texture.wetness)
    eff_bluff_f = max(0.0, min(0.50, bluff_f * texture_bluff_scale * rev_bluff_sc))

    # --- Step 7: Compute bet sizing fraction for this action ---
    strong  = eff_eq >= value_t
    sz_frac = street_sizing(street, texture, strong) * rev_size_sc

    # Draw vs. made hand sizing distinction:
    # - Semi-bluffs with draws: use smaller sizing (risk less with incomplete hand).
    # - Made hands on wet boards: use larger sizing (charge opponents drawing against us).
    if is_draw and not strong:
        sz_frac *= 0.80
    elif not is_draw and strong and texture.wetness > 0.35:
        sz_frac *= 1.15

    sz_frac = max(0.25, min(1.40, sz_frac))

    # --- Step 7 (alt): Jam/Fold shortcut for very low SPR situations ---
    # When the effective stack is less than 3× the pot, we are pot-committed.
    # In this case, skip nuanced sizing: jam if ahead, fold/check if behind.
    if spr < SPR_LOW and street != 'pre-flop':
        if eff_eq >= 0.60:
            # We are a slight favorite or better → go all-in / call.
            if eff_eq < 0.64 and state.can_act(ActionCall):
                return ActionCall()
            elif state.can_act(ActionRaise):
                return ActionRaise(max_raise)
            elif state.can_act(ActionCall):
                return ActionCall()
        else:
            # We are behind → get out or check if check is free.
            if state.can_act(ActionCheck):
                return ActionCheck()
            elif state.can_act(ActionFold):
                return ActionFold()

    # =========================================================================
    # ROUTE 1: We can check (no bet to face / we are the aggressor).
    # =========================================================================
    if state.can_act(ActionCheck):

        # --- Check-raise trap ---
        # Against aggressive opponents, occasionally check strong hands to
        # induce a bet, then reraise on the next action (ROUTE 2 handles it).
        if (eff_eq >= value_t
                and street in ('flop', 'turn')
                and strategy.model.sufficient_data()
                and strategy.model.postflop_aggression >= CHECK_RAISE_MIN_AGG
                and random.random() < CHECK_RAISE_FREQ):
            return ActionCheck()

        # --- River-specific polarized logic ---
        if street == 'river':
            if eff_eq >= value_t:
                # Strong hand: polarized large value bet.
                if state.can_act(ActionRaise):
                    river_sz = RIVER_VALUE_SIZE * rev_size_sc
                    return ActionRaise(clamp(pot + int(pot * river_sz)))
                return ActionCheck()
            elif eff_eq < fold_t and random.random() < eff_bluff_f * RIVER_BLUFF_REDUCTION:
                # River bluff: reduced frequency, same large sizing as value (GTO polarized).
                if state.can_act(ActionRaise):
                    river_sz = RIVER_VALUE_SIZE * rev_size_sc
                    return ActionRaise(clamp(pot + int(pot * river_sz)))
                return ActionCheck()
            else:
                # Marginal hands and failed bluff checks: just check.
                return ActionCheck()

        # Strong made hand → fire a full value bet (non-river streets).
        if eff_eq >= value_t:
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * sz_frac)))
            return ActionCheck()

        # Marginal value hand → make a thin bet to build the pot.
        elif eff_eq >= call_t:
            thin_sz = street_sizing(street, texture, False) * rev_size_sc
            thin_sz = max(0.25, min(1.40, thin_sz))
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * thin_sz)))
            return ActionCheck()

        # Weak hand → stochastic bluff or give up with a check.
        else:
            if (street in ('flop', 'turn')
                    and eff_eq < fold_t
                    and random.random() < eff_bluff_f
                    and state.can_act(ActionRaise)):
                bluff_sz = street_sizing(street, texture, False) * rev_size_sc
                bluff_sz = max(0.25, min(1.40, bluff_sz))
                return ActionRaise(clamp(pot + int(pot * bluff_sz)))
            return ActionCheck()

    # =========================================================================
    # ROUTE 2: Facing a bet from the opponent.
    # =========================================================================

    else:
        # --- Pre-flop near-all-in: shove/fold decision ---
        if ctc >= state.my_chips * 0.8 and street == 'pre-flop':
            shove_range = estimate_shove_range(strategy.model)
            eff_eq = monte_carlo_equity(hole, [], [], MC_SIMS_AUCTION, shove_range)
            return ActionCall() if eff_eq > 0.54 else ActionFold()

        # --- River-specific adjustments when facing a bet ---
        if street == 'river':
            # vs passive on river: lower call threshold (exploit value-heavy bets).
            # vs aggressive on river: also lower call threshold (pick off bluffs).
            if strategy.model.sufficient_data() and strategy.model.is_passive():
                call_t_river = max(call_t - 0.04, 0.30)
            elif strategy.model.sufficient_data() and strategy.model.is_aggressive():
                call_t_river = max(call_t - 0.03, 0.32)
            else:
                call_t_river = call_t

            if eff_eq >= value_t:
                if state.can_act(ActionRaise):
                    return ActionRaise(clamp(int((pot + ctc) * 0.81)))
                if state.can_act(ActionCall):
                    return ActionCall()
            elif eff_eq >= max(call_t_river, pot_odds * 1.10):
                if state.can_act(ActionCall):
                    return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
        else:
            # Non-river streets: original logic.
            if eff_eq >= value_t:
                if state.can_act(ActionRaise):
                    return ActionRaise(clamp(int((pot + ctc) * 0.81)))
                if state.can_act(ActionCall):
                    return ActionCall()
            elif eff_eq >= max(call_t, pot_odds * 1.10):
                if state.can_act(ActionCall):
                    return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()

    # =========================================================================
    # FALLBACK: Should never reach here under normal circumstances.
    # These guards prevent the bot from crashing if all primary paths fail.
    # =========================================================================
    if state.can_act(ActionCall):   return ActionCall()
    if state.can_act(ActionCheck):  return ActionCheck()
    return ActionFold()


# =============================================================================
# SECTION 15: PLAYER (MAIN BOT CONTROLLER)
#
# The Player class is the integration point between the game engine and the
# internal strategy/model logic. The engine calls:
#   - on_hand_start()  at the beginning of every new hand
#   - on_hand_end()    after a hand finishes
#   - get_move()       whenever the bot must take an action
#
# Player creates OpponentModel and AdaptiveStrategy once at construction,
# so they persist across the entire session and accumulate data.
# =============================================================================

class Player(BaseBot):
    """
    Main bot controller. Manages session-level state (OpponentModel,
    AdaptiveStrategy) and hand-level transient state, routing observations
    and actions through the appropriate subsystems.
    """

    def __init__(self) -> None:
        """
        Initialize session-persistent objects and reset all transient
        hand-level variables.

        OpponentModel and AdaptiveStrategy are created once here and
        live for the entire session. All other attributes are reset
        at the start of each hand by on_hand_start().
        """
        # Session-persistent: these accumulate data across all hands.
        self.opp_model  = OpponentModel()
        self.strategy   = AdaptiveStrategy(self.opp_model)

        self._street_reraise_count  = 0   # Opponent reraises seen on the current street
        self._last_seen_street      = None  # Track street transitions for reset

        # Transient hand-level variables — reset by on_hand_start() each hand.
        self._pf_recorded       = False   # Flag: have we logged pre-flop action?
        self._streets_seen      = set()   # Streets we have observed this hand
        self._opp_bet_streets   = set()   # Streets on which opponent bet/raised
        self._our_auction_bid   = 0       # The bid we placed this hand (0 if none)
        self._is_bb             = False   # Are we the Big Blind this hand?
        self._we_won_auction    = False   # Did we win the auction reveal?
        self._revealed_card     = None    # The specific card revealed to us (or None)
        self._opp_won_auction   = False   # Did the opponent win the auction?
        self._opp_bet_post_auction = False  # Did opp bet the flop after winning?
        self._auction_seen      = False   # Have we processed the auction outcome?

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        """
        Reset all hand-level transient state at the beginning of a new hand.

        Called by the engine before the first action of every hand. Without
        this reset, data from the previous hand would bleed into the new one.

        Args:
            game_info:     Static session metadata (stack sizes, etc.).
            current_state: Initial state of the new hand.
        """
        self._pf_recorded          = False
        self._streets_seen         = set()
        self._opp_bet_streets      = set()
        self._our_auction_bid      = 0
        self._is_bb                = current_state.is_bb   # Determined by seat this hand
        self._we_won_auction       = False
        self._revealed_card        = None
        self._opp_won_auction      = False
        self._opp_bet_post_auction = False
        self._auction_seen         = False
        self._street_reraise_count  = 0
        self._last_seen_street      = None

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """
        Commit all observations from the completed hand to the OpponentModel.

        Called by the engine after a hand finishes (fold, showdown, or all-in
        runout). This is the only place where we write to the persistent model.

        Records:
            - Post-flop street betting behavior (per street observed).
            - Our auction bid (if we placed one).
            - Auction outcome (who won, did winner bet flop).
            - Final hand result (payoff, last street, opponent cards shown).

        Args:
            game_info:     Static session metadata.
            current_state: Final state after the hand concluded.
        """
        # Record per-street betting observations: was each street seen, and
        # did the opponent bet on it?
        for street in ('flop', 'turn', 'river'):
            if street in self._streets_seen:
                self.opp_model.record_street_bet(
                    street, street in self._opp_bet_streets)

        # Record our auction bid if we participated in the auction this hand.
        if self._our_auction_bid > 0:
            self.opp_model.record_auction(self._our_auction_bid)

        # Retrieve any cards the opponent revealed at showdown.
        opp_revealed = (list(current_state.opp_revealed_cards)
                        if current_state.opp_revealed_cards else [])

        # Record auction outcome if an auction occurred this hand.
        if self._auction_seen:
            self.opp_model.record_opp_auction_result(
                opp_won      = self._opp_won_auction,
                opp_bet_after = self._opp_bet_post_auction,
            )

        # Record the final hand outcome: payoff, last street, and showdown cards.
        self.opp_model.record_hand_end(
            payoff       = current_state.payoff,
            final_street = current_state.street,
            opp_revealed = opp_revealed,
        )

    def _observe_preflop(self, state: PokerState):
        """
        Detect pre-flop aggression and log it to the opponent model.

        Called every action on the pre-flop street, but uses _pf_recorded
        to ensure we only log once per hand. If the cost to call is more
        than the standard BB-SB differential, the opponent raised.

        Args:
            state: Current game state.
        """
        if self._pf_recorded or state.street != 'pre-flop':
            return   # Already recorded or wrong street

        # cost_to_call > (BB - SB) means the opponent raised above the initial bet.
        was_aggressive = state.cost_to_call > (BIG_BLIND - SMALL_BLIND)
        self.opp_model.record_preflop(was_aggressive)
        self._pf_recorded = True

    def _observe_postflop(self, state: PokerState):
        """
        Track which post-flop streets we see and whether the opponent bets
        on each street. This data feeds into the post-flop aggression rate.

        A positive cost_to_call on a post-flop street means the opponent bet
        or raised before this action was requested.

        Args:
            state: Current game state.
        """
        s = state.street
        if s not in ('flop', 'turn', 'river'):
            return   # Pre-flop and auction streets are handled separately

        if state.cost_to_call > 0:
            self._opp_bet_streets.add(s)   # Opponent bet on this street
        self._streets_seen.add(s)           # We witnessed this street

    def _observe_auction_outcome(self, state: PokerState):
        """
        Determine who won the auction by examining the game state immediately
        after the auction phase.

        Called the first time we see the flop (which follows the auction).
        Only processes once per hand via the _auction_seen guard.

        Detection logic:
            - If opp_revealed_cards is populated → we won (we see their card).
            - Else if we placed a non-zero bid → opponent won (we bid but got nothing).

        Args:
            state: Current game state (at the start of the flop).
        """
        if self._auction_seen:
            return   # Only process auction outcome once per hand

        self._auction_seen = True
        opp_revealed = (list(state.opp_revealed_cards)
                        if state.opp_revealed_cards else [])

        if opp_revealed:
            # We can see the opponent's card → we won the auction.
            self._we_won_auction = True
            self._revealed_card  = opp_revealed[0]   # Store for use in decide_action()
            self._opp_won_auction = False

        elif self._our_auction_bid > 0:
            # We bid but received no reveal → opponent outbid us.
            self._opp_won_auction = True

    def _observe_opp_post_auction_bet(self, state: PokerState):
        """
        Detect whether the opponent bets on the flop immediately after winning
        the auction. This flag feeds into the post-auction aggression metric
        in OpponentModel.

        Uses a sentinel value ('flop_post_auction_checked') in _streets_seen
        to ensure this check runs only once per hand.

        Args:
            state: Current game state (flop only).
        """
        if (self._opp_won_auction
                and state.street == 'flop'
                and 'flop_post_auction_checked' not in self._streets_seen):

            # Mark as checked so this block doesn't run again this hand.
            self._streets_seen.add('flop_post_auction_checked')

            # If there's a bet facing us on the flop, opponent used their auction info.
            if state.cost_to_call > 0:
                self._opp_bet_post_auction = True

    def _observe_reraises(self, state: PokerState):
        """
        Track how many times the opponent has reraised on the current street.
        Resets the counter whenever the street changes.

        A reraise is detected when cost_to_call exceeds the previous cost_to_call
        on the same street — meaning the opponent raised on top of our last action.
        """
        s = state.street

        # Street changed → reset counter and last-call tracking
        if s != self._last_seen_street:
            self._street_reraise_count = 0
            self._last_ctc             = 0
            self._last_seen_street     = s

        # A reraise: opponent raised again after we had already faced a bet
        if state.cost_to_call > 0 and state.cost_to_call > getattr(self, '_last_ctc', 0):
            self._street_reraise_count += 1

        self._last_ctc = state.cost_to_call


    def get_move(self, game_info: GameInfo, current_state: PokerState) \
            -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        """
        Main decision entry point, called by the game engine on every action.

        Orchestrates the full decision pipeline:
            1. Observe and log pre-flop/post-flop context.
            2. Retrieve opponent range inference.
            3. Handle auction bidding as a special-case street.
            4. On the flop, resolve and cache auction outcomes.
            5. Compute equity via MC simulation (or table lookup pre-flop).
            6. Analyze board texture.
            7. Bundle auction context into a dict for decide_action().
            8. Delegate to decide_action() and return the result.

        Args:
            game_info:     Static session metadata.
            current_state: Current game state requiring an action.

        Returns:
            An action object: ActionFold, ActionCall, ActionCheck,
            ActionRaise, or ActionBid.
        """
        # Step 1: Update pre-flop and post-flop observation trackers.
        self._observe_preflop(current_state)
        self._observe_postflop(current_state)
        self._observe_reraises(current_state)

        # Step 2: Retrieve inferred opponent range from the model.
        # This fraction constrains the MC simulation's opponent hand sampling.
        opp_rf = self.opp_model.inferred_range_fraction()

        # Step 3: Auction street — compute and return a bid immediately.
        # This bypasses all equity/texture logic since bidding is entirely
        # different from normal bet/raise/call/fold decisions.
        if current_state.street == 'auction':
            bid = compute_bid(current_state, self.strategy, opp_rf)
            self._our_auction_bid = bid   # Cache bid for on_hand_end() tracking
            return ActionBid(bid)

        # Step 4: Flop — resolve who won the auction before making decisions.
        # Also check if the opponent immediately bet after winning.
        if current_state.street == 'flop':
            self._observe_auction_outcome(current_state)
            self._observe_opp_post_auction_bet(current_state)

        # Step 5: Compute equity for the current hand/board state.
        equity = get_equity(current_state, MC_SIMS, opp_rf)

        # Step 6: Analyze board texture (wetness, flush draws, etc.).
        board   = list(current_state.board) if current_state.board else []
        texture = board_texture(board)

        # Step 7: Bundle auction context into a clean dict for decide_action().
        auction_ctx = {
            'we_won_auction':       self._we_won_auction,
            'revealed_card':        self._revealed_card,
            'opp_won_auction':      self._opp_won_auction,
            'opp_bet_post_auction': self._opp_bet_post_auction,
            'is_bb':                self._is_bb,
            'street_reraise_count':  self._street_reraise_count,
        }

        # Step 8: Delegate to the core decision engine and return the action.
        return decide_action(current_state, equity, self.strategy, texture, auction_ctx)


# =============================================================================
# SECTION 16: ENTRY POINT
#
# Connects the Player bot to the game server runner when executed directly.
# parse_args() reads command-line arguments (host, port, etc.);
# run_bot() starts the client loop, calling get_move() for each action.
# =============================================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())# post-flop), information from the auction is more valuable because the
# opponent will always have positional advantage. This multiplier increases
# the raw computed bid by 40% to compensate.
OOP_BID_MULTIPLIER = 1.4

# Number of Monte Carlo simulation iterations used for post-flop equity
# estimation. Higher values = more accurate but slower. 1000 is a good
# balance for real-time play.
MC_SIMS = 1600

# Separate (lower) simulation count used during the auction phase only.
# Kept lower because compute_bid() already runs multiple equity calculations
# in parallel, and we need a response quickly.
MC_SIMS_AUCTION = 200

# Equity thresholds that gate different actions:
#   FOLD_THRESHOLD  — below this, prefer folding when facing a bet
#   CALL_THRESHOLD  — minimum equity needed to call a bet
#   VALUE_THRESHOLD — minimum equity required to fire a value bet/raise
FOLD_THRESHOLD  = 0.35
CALL_THRESHOLD  = 0.42
VALUE_THRESHOLD = 0.65

# Probability (0.0–1.0) that the bot attempts a bluff when it would otherwise
# check. Applied stochastically so bluffing isn't predictable.
BLUFF_FREQ = 0.16

# Default bet sizing fractions relative to the pot.
# These are starting points; they get adjusted by street and board texture.
BET_SMALL_FRAC = 0.50   # Half-pot sizing for thin value / merged ranges
BET_BIG_FRAC   = 0.75   # Three-quarter pot for strong value hands

# Minimum hand history before the opponent model's statistics are trusted.
# Before this many hands, all thresholds fall back to baseline defaults.
MODEL_MIN_HANDS = 40

# Stack-to-Pot Ratio (SPR) boundary values:
#   SPR < SPR_LOW  → players are pot-committed, simplify to jam/fold
#   SPR > SPR_HIGH → deep stacks, drawing hands gain implied odds value
SPR_LOW  = 3.0
SPR_HIGH = 9.0

# Range fractions map to the top X% of all possible starting hands,
# sorted by pre-flop equity. Used to constrain MC opponent hand sampling.
RANGE_TIGHT  = 0.30   # Only the top 30% of hands (tight/nit opponent)
RANGE_MEDIUM = 0.60   # Top 60% of hands (average opponent)
RANGE_LOOSE  = 0.96   # All hands (no range constraint applied)

# Integer rank thresholds used when interpreting a card revealed during the
# auction phase. Rank values: 2=2, ..., 10=T, 11=J, 12=Q, 13=K, 14=A.
REVEALED_HIGH_RANK = 11   # Jack or above → opponent has a strong card
REVEALED_LOW_RANK  = 7    # 7 or below → opponent has a weak card

# When the opponent's revealed auction card is high (Jack+):
#   - Reduce our equity estimate slightly (opponent has a good card)
#   - Reduce bluff frequency (harder to fold out a strong hand)
#   - Reduce bet sizing (we lose value betting into a good range)
REVEALED_HIGH_EQ_DELTA    = -0.04
REVEALED_HIGH_BLUFF_SCALE =  0.70
REVEALED_HIGH_SIZE_SCALE  =  0.85

# When the revealed card is low (7 or below):
#   - Increase equity slightly (opponent has a weak card)
#   - Increase bluff frequency (easier to fold them out)
#   - Increase bet sizing (we get more value)
REVEALED_LOW_EQ_DELTA    = 0.02
REVEALED_LOW_BLUFF_SCALE = 1.10
REVEALED_LOW_SIZE_SCALE  = 1.15

# Dynamic ceiling for how large our auction bid can be as a fraction of the
# pot. The range [BID_CAP_MIN, BID_CAP_MAX] is scaled based on pot size and
# stack depth inside dynamic_max_bid_frac().
BID_CAP_MIN = 0.16   # Never bid below 16% of pot
BID_CAP_MAX = 0.60   # Never bid above 60% of pot

# When the opponent wins the auction (they see one of their cards), the bot
# plays more conservatively. AUCTION_LOSS_K is a multiplier close to 1.0 that
# gently tightens bluff frequency and shifts equity thresholds upward.
# Value of 0.84 means a 16% tightening effect.
AUCTION_LOSS_K = 0.84

# Minimum auction wins before we fully trust revealed card adjustments.
# Below this, adjustments are dampened proportionally.
AUCTION_CONFIDENCE_MIN = 16

# Thresholds for interpreting the opponent's flop-betting behavior after they
# win the auction. A high post-auction bet rate signals that the revealed card
# genuinely strengthened their hand or they are very aggressive.
OPP_AUCT_AGG_HIGH  = 0.65   # ≥65% → opponent almost always bets with auction card
OPP_AUCT_AGG_LOW   = 0.30   # ≤30% → opponent often traps or slow-plays
OPP_AUCT_FOLD_BUMP = 0.07   # Raise fold threshold by this much vs. aggressive post-auction bettors
OPP_AUCT_CALL_DIP  = 0.05   # Lower call threshold by this much vs. passive post-auction players

# Equity and bluff adjustments based on the bot's table position.
# In position (SB / not BB): slight equity bonus since we act last post-flop.
# Out of position (BB): slight penalty since we are at an informational disadvantage.
POSITION_IP_EQ_BONUS    =  0.02   # Add to equity when in position (SB)
POSITION_OOP_EQ_DELTA   = -0.02   # Subtract from equity when out of position (BB)
POSITION_IP_BLUFF_SCALE =  1.10   # Multiply bluff frequency by 1.1 when in position
POSITION_OOP_BLUFF_SCALE =  0.90   # Multiply bluff frequency by 0.9 when out of position
POSITION_BB_CALL_BONUS  =  0.04   # Lower the call threshold by 0.04 for BB (close the action discount)

# When facing an aggressive opponent with a strong hand, occasionally check
# to induce a bet and then reraise (check-raise trap).
CHECK_RAISE_FREQ    = 0.28   # Probability of trapping with a strong hand vs aggro opponent
CHECK_RAISE_MIN_AGG = 0.40   # Minimum opponent postflop_aggression to trigger trap

# River-specific polarized sizing: bet big with value and bluffs (GTO polarization).
RIVER_VALUE_SIZE      = 0.85  # Pot fraction for river value bets (polarized large)
RIVER_BLUFF_REDUCTION = 0.60  # Multiply bluff frequency by this on the river


# =============================================================================
# SECTION 2: CARD UTILITY PRIMITIVES
#
# Basic helpers for parsing card strings like 'Ah', 'Td', '2c'.
# =============================================================================

# All valid card ranks in ascending order (2 through Ace).
RANKS = '23456789TJQKA'

# All four suits in no particular order.
SUITS = 'hdcs'

# Maps rank characters to integer values (e.g., '2'→2, 'T'→10, 'A'→14).
# enumerate(..., 2) starts the counter at 2 so '2'→2 naturally.
RANK_VAL = {r: i for i, r in enumerate(RANKS, 2)}

# Reverse mapping: integer rank -> display character. Used by _hand_key().
_RD = {v: k for k, v in RANK_VAL.items()}

# Complete 52-card deck as a flat list of strings, e.g., ['2h','2d',...,'As','Ac'].
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def card_rank(card: str) -> int:
    """
    Extract the integer rank value from a card string.

    Example:
        card_rank('Ah') → 14
        card_rank('Tc') → 10
        card_rank('2d') →  2
    """
    return RANK_VAL[card[0]]


def card_suit(card: str) -> str:
    """
    Extract the suit character from a card string.

    Example:
        card_suit('Ah') → 'h'
        card_suit('2c') → 'c'
    """
    return card[1]


# =============================================================================
# SECTION 3: HAND EVALUATION WRAPPERS (eval7 library)
#
# eval7 is a fast Cython-based poker hand evaluator.
# Higher return values mean stronger hands.
# =============================================================================

def convert_to_cards(cards):
    """
    Convert a list of card strings (e.g., ['Ah', 'Kd']) into eval7.Card objects.
    eval7 requires its own Card type for hand evaluation.
    """
    return [eval7.Card(str(c)) for c in cards]


# Module-level card-string -> eval7.Card cache. Avoids repeated object creation
# inside hot MC loops. There are only 52 possible card strings.
_E7_CACHE: dict[str, eval7.Card] = {}


def _e7(card_str: str) -> eval7.Card:
    """Return the cached eval7.Card for a given card string like 'Ah'."""
    c = _E7_CACHE.get(card_str)
    if c is None:
        c = eval7.Card(card_str)
        _E7_CACHE[card_str] = c
    return c


# Pre-populate the cache for all 52 cards at import time.
for _cs in FULL_DECK:
    _e7(_cs)


def best_hand_score(hole: list, board: list) -> int:
    """
    Evaluate the best possible 5-card hand from a player's hole cards and
    any number of community cards (can be 3, 4, or 5 board cards).

    eval7.evaluate() automatically selects the best 5-card combination
    from all cards provided.

    Args:
        hole:  The player's 2 private hole cards.
        board: The community cards visible to all players.

    Returns:
        Integer strength score (higher = better).
    """
    all_cards = convert_to_cards(hole + board)
    return eval7.evaluate(all_cards)


# =============================================================================
# SECTION 4: PRE-FLOP EQUITY LOOKUP TABLE
#
# Running Monte Carlo simulations pre-flop on every action would be too slow
# and is unnecessary since pre-flop equities are static. This table stores
# the win probability for every canonical heads-up starting hand versus a
# random opponent hand.
#
# Keys:
#   'AA', 'KK', ...         → Pocket pairs (always 2 chars)
#   'AKs', 'QTs', ...       → Suited unpaired hands ('s' suffix)
#   'AKo', 'Q8o', ...       → Offsuit unpaired hands ('o' suffix)
# Values:
#   Float representing win probability (e.g., 0.853 = 85.3% vs. random hand)
# =============================================================================

_PF_EQUITY: dict[str, float] = {
    # Pocket Pairs
    'AA': 0.852, 'KK': 0.824, 'QQ': 0.799, 'JJ': 0.775, 'TT': 0.751,
    '99': 0.716, '88': 0.689, '77': 0.662, '66': 0.633, '55': 0.603,
    '44': 0.570, '33': 0.537, '22': 0.503,

    # Suited Aces
    'AKs': 0.670, 'AQs': 0.661, 'AJs': 0.654, 'ATs': 0.647, 'A9s': 0.628,
    'A8s': 0.619, 'A7s': 0.609, 'A6s': 0.599, 'A5s': 0.599, 'A4s': 0.588,
    'A3s': 0.579, 'A2s': 0.569,

    # Offsuit Aces
    'AKo': 0.653, 'AQo': 0.644, 'AJo': 0.635, 'ATo': 0.627, 'A9o': 0.607,
    'A8o': 0.596, 'A7o': 0.584, 'A6o': 0.573, 'A5o': 0.573, 'A4o': 0.561,
    'A3o': 0.551, 'A2o': 0.540,

    # Suited Kings
    'KQs': 0.633, 'KJs': 0.625, 'KTs': 0.617, 'K9s': 0.599, 'K8s': 0.584,
    'K7s': 0.576, 'K6s': 0.567, 'K5s': 0.557, 'K4s': 0.547, 'K3s': 0.537,
    'K2s': 0.528,

    # Offsuit Kings
    'KQo': 0.614, 'KJo': 0.606, 'KTo': 0.597, 'K9o': 0.577, 'K8o': 0.560,
    'K7o': 0.551, 'K6o': 0.541, 'K5o': 0.530, 'K4o': 0.518, 'K3o': 0.507,
    'K2o': 0.497,

    # Suited Queens
    'QJs': 0.602, 'QTs': 0.594, 'Q9s': 0.575, 'Q8s': 0.559, 'Q7s': 0.545,
    'Q6s': 0.536, 'Q5s': 0.525, 'Q4s': 0.514, 'Q3s': 0.504, 'Q2s': 0.495,

    # Offsuit Queens
    'QJo': 0.582, 'QTo': 0.573, 'Q9o': 0.552, 'Q8o': 0.535, 'Q7o': 0.519,
    'Q6o': 0.508, 'Q5o': 0.497, 'Q4o': 0.484, 'Q3o': 0.473, 'Q2o': 0.463,

    # Suited Jacks
    'JTs': 0.575, 'J9s': 0.555, 'J8s': 0.539, 'J7s': 0.525, 'J6s': 0.508,
    'J5s': 0.497, 'J4s': 0.485, 'J3s': 0.475, 'J2s': 0.465,

    # Offsuit Jacks
    'JTo': 0.553, 'J9o': 0.531, 'J8o': 0.513, 'J7o': 0.497, 'J6o': 0.479,
    'J5o': 0.466, 'J4o': 0.453, 'J3o': 0.442, 'J2o': 0.431,

    # Suited Tens
    'T9s': 0.540, 'T8s': 0.523, 'T7s': 0.508, 'T6s': 0.491, 'T5s': 0.474,
    'T4s': 0.461, 'T3s': 0.450, 'T2s': 0.440,

    # Offsuit Tens
    'T9o': 0.516, 'T8o': 0.497, 'T7o': 0.480, 'T6o': 0.461, 'T5o': 0.442,
    'T4o': 0.427, 'T3o': 0.415, 'T2o': 0.403,

    # Suited Nines
    '98s': 0.508, '97s': 0.492, '96s': 0.474, '95s': 0.459, '94s': 0.441,
    '93s': 0.430, '92s': 0.419,

    # Offsuit Nines
    '98o': 0.481, '97o': 0.463, '96o': 0.443, '95o': 0.426, '94o': 0.405,
    '93o': 0.393, '92o': 0.380,

    # Suited Eights
    '87s': 0.479, '86s': 0.461, '85s': 0.445, '84s': 0.428, '83s': 0.410,
    '82s': 0.399,

    # Offsuit Eights
    '87o': 0.450, '86o': 0.430, '85o': 0.412, '84o': 0.392, '83o': 0.372,
    '82o': 0.358,

    # Suited Sevens
    '76s': 0.457, '75s': 0.440, '74s': 0.423, '73s': 0.404, '72s': 0.385,

    # Offsuit Sevens
    '76o': 0.426, '75o': 0.407, '74o': 0.387, '73o': 0.366, '72o': 0.346,

    # Suited Sixes
    '65s': 0.438, '64s': 0.421, '63s': 0.402, '62s': 0.381,

    # Offsuit Sixes
    '65o': 0.406, '64o': 0.386, '63o': 0.365, '62o': 0.342,

    # Suited Fives
    '54s': 0.427, '53s': 0.409, '52s': 0.389,

    # Offsuit Fives
    '54o': 0.394, '53o': 0.373, '52o': 0.351,

    # Suited Fours
    '43s': 0.400, '42s': 0.380,

    # Offsuit Fours
    '43o': 0.364, '42o': 0.341,

    # Suited Threes
    '32s': 0.365,

    # Offsuit Threes
    '32o': 0.323
}


@lru_cache(maxsize=2704)
def _hand_key(c1: str, c2: str) -> str:
    """
    Normalize two hole cards into a canonical lookup key for _PF_EQUITY.

    The key format follows standard poker shorthand:
        - Pocket pairs:     'AA', 'KK', '22', etc.
        - Suited hands:     'AKs', 'QTs', etc.  (higher rank first)
        - Offsuit hands:    'AKo', 'J9o', etc.  (higher rank first)

    Uses lru_cache because the same card pair can appear hundreds of times
    across a long session, and string construction is surprisingly expensive.

    Args:
        c1, c2: Two card strings, e.g., 'Ah', 'Kd'

    Returns:
        A string key that matches a key in _PF_EQUITY.
    """
    r1, r2 = card_rank(c1), card_rank(c2)
    s1, s2 = card_suit(c1), card_suit(c2)

    # Always put the higher-rank card first in the key string.
    hi_r, lo_r = max(r1, r2), min(r1, r2)

    # Pocket pairs: same rank, no suit suffix needed.
    if hi_r == lo_r:
        return _RD[hi_r] + _RD[lo_r]

    # Unpaired hands: append 's' for suited, 'o' for offsuit.
    suffix = 's' if s1 == s2 else 'o'
    return _RD[hi_r] + _RD[lo_r] + suffix


def preflop_equity(hole: list) -> float:
    """
    Return the pre-flop win probability for our hole cards against a random
    opponent hand in heads-up play.

    Primary path: direct lookup in _PF_EQUITY via the canonical key.

    Default fallback: if the key is somehow missing (should not happen), return 0.5 as a neutral equity guess.
    
    Args:
        hole: List of exactly 2 card strings.

    Returns:
        Float in [0.0, 1.0] representing our win probability pre-flop.
    """
    key = _hand_key(hole[0], hole[1])
    if key in _PF_EQUITY:
        return _PF_EQUITY[key]

    return 0.5  # Default fallback equity if key is missing (should not occur)

# =============================================================================
# SECTION 5: OPPONENT RANGE CONSTRUCTION
#
# We constrain the Monte Carlo simulations so that the opponent is only ever
# assigned hands from within their estimated "range" (the set of hands they
# would actually play). This makes equity estimates more realistic than
# simulating vs. all 52 cards randomly.
# =============================================================================

# Pre-sort all hand keys from strongest to weakest pre-flop equity.
# This is the "range stack": the top N% of this list forms an opponent's range.
_ALL_HAND_KEYS_SORTED: list[str] = sorted(
    _PF_EQUITY.keys(), key=lambda k: _PF_EQUITY[k], reverse=True
)

# Pre-build the mapping from canonical hand key → all physical card pairs
# for that hand class. For example:
#   'AA' → [('Ah','Ad'), ('Ah','Ac'), ('Ah','As'), ('Ad','Ac'), ...]
#   'AKs'→ [('Ah','Kh'), ('Ad','Kd'), ('Ac','Kc'), ('As','Ks')]
#   'AKo'→ all 12 offsuit combinations of an Ace and a King
#
# Computing this at import time avoids repeated work during simulations.
_CARDS_FOR_KEY_FULL: dict[str, list] = {}

for _key in _ALL_HAND_KEYS_SORTED:
    if len(_key) == 2:
        # Pocket pair: both cards share the same rank, different suits.
        # We use _s1 < _s2 to avoid duplicate pairs like ('Ah','Ad') and ('Ad','Ah').
        _r = _key[0]
        _CARDS_FOR_KEY_FULL[_key] = [(_r + _s1, _r + _s2)
                                      for _s1 in SUITS for _s2 in SUITS
                                      if _s1 < _s2]
    else:
        # Unpaired hand: separate high rank, low rank, and suited flag.
        _hi_r, _lo_r, _suited = _key[0], _key[1], _key[2] == 's'
        _pairs = []
        for _s1 in SUITS:
            for _s2 in SUITS:
                # For suited hands: both cards must share a suit.
                if _suited and _s1 != _s2:
                    continue
                # For offsuit hands: suits must differ.
                if not _suited and _s1 == _s2:
                    continue
                _c1, _c2 = _hi_r + _s1, _lo_r + _s2
                # Sanity guard: a card cannot be paired with itself.
                if _c1 != _c2:
                    _pairs.append((_c1, _c2))
        _CARDS_FOR_KEY_FULL[_key] = _pairs


def _cards_for_key(key: str, excluded: set) -> list:
    """
    Return all physical card pairs for a given hand class that do not contain
    any cards already in use (in our hand, on the board, or in opp_known).

    Args:
        key:      Canonical hand key like 'AKs', 'QQ'.
        excluded: Set of card strings that cannot be dealt.

    Returns:
        List of (card1, card2) tuples still available to be assigned.
    """
    return [(a, b) for a, b in _CARDS_FOR_KEY_FULL[key]
            if a not in excluded and b not in excluded]


@lru_cache(maxsize=256)
def _build_opponent_range_cached(range_fraction: float,
                                  excluded_frozen: frozenset) -> tuple:
    """
    Build and cache the set of valid opponent hand pairs for a given range
    fraction and set of excluded (dead) cards.

    The range is constructed by taking the top `range_fraction * N` hand
    types from _ALL_HAND_KEYS_SORTED and expanding each to concrete card pairs,
    filtering out dead cards.

    Args:
        range_fraction:   Float in (0, 1.0] representing the fraction of all
                          hand types to include (e.g., 0.25 = top 25%).
        excluded_frozen:  frozenset of unavailable card strings (must be
                          hashable for lru_cache).

    Returns:
        Tuple of (card1, card2) pairs making up the opponent's playable range.
    """
    n_types  = max(1, int(len(_ALL_HAND_KEYS_SORTED) * range_fraction))
    top_keys = _ALL_HAND_KEYS_SORTED[:n_types]
    pairs    = []
    for key in top_keys:
        pairs.extend(_cards_for_key(key, excluded_frozen))
    return tuple(pairs)


def build_opponent_range(range_fraction: float, excluded: set) -> list:
    """
    Public wrapper around _build_opponent_range_cached that accepts a regular
    set (which is not hashable) by converting it to a frozenset internally.

    Args:
        range_fraction: Fraction of all hand types to include.
        excluded:       Set of dead card strings.

    Returns:
        List of valid (card1, card2) opponent hand pairs.
    """
    return list(_build_opponent_range_cached(range_fraction, frozenset(excluded)))


# =============================================================================
# SECTION 6: MONTE CARLO EQUITY ESTIMATION
#
# For post-flop decisions we cannot rely on static tables — the board changes
# the relative strength of every hand. Instead, we run N random "runouts"
# (deal out the remaining board cards and a random opponent hand) and count
# how often our hand wins.
#
# This section handles two cases:
#   Case 1: One opponent card is known; the other is sampled from their range.
#   Case 2: Neither opponent card is known; both drawn randomly from range.
# =============================================================================

def monte_carlo_equity(hole: list, board: list, opp_known: list,
                       n_sims: int = MC_SIMS,
                       opp_range_fraction: float = RANGE_LOOSE) -> float:
    """
    Estimate our win probability via Monte Carlo simulation.

    For each simulation iteration:
        1. Sample a valid opponent hand (respecting known cards and their range).
        2. Complete the community board to 5 cards.
        3. Evaluate both hands and record win/tie/loss.
    Returns the average win rate (ties count as 0.5 wins).

    Args:
        hole:               Our 2 hole cards.
        board:              Current community cards (0–5 cards).
        opp_known:          Cards we know the opponent holds (0, 1, or 2).
                            Usually 1 card if we won the auction.
        n_sims:             Number of Monte Carlo iterations.
        opp_range_fraction: Fraction of all hand types the opponent plays.
                            Lower = tighter range = stronger opponent hands on average.

    Returns:
        Float in [0.0, 1.0] representing our estimated win probability.
    """
    # Build the set of all cards that are already "in play" and cannot be dealt.
    known_set = set(hole + board + opp_known)
    remaining = [c for c in FULL_DECK if c not in known_set]

    # Determine how many additional cards we need to complete the board and
    # how many of the opponent's hole cards are still unknown.
    cards_needed_board = 5 - len(board)
    cards_needed_opp   = 2 - len(opp_known)

    # Pre-build the opponent range if their hand isn't fully known and we're
    # not using a completely unconstrained (random) range.
    range_pairs = build_opponent_range(opp_range_fraction, known_set)
    second_card_pool = None  # Will hold the valid second card options if one opponent card is known.
    if cards_needed_opp == 1:
        # If one opponent card is known, filter the range to only pairs that include that card.
        known_card = opp_known[0]
        second_card_pool = list({
                c
                for pair in range_pairs
                for c in pair
                if c != known_card and c not in known_set
            })

    # Pre-convert known cards to eval7 objects ONCE before the loop.
    e7_hole = [_e7(c) for c in hole]
    e7_board = [_e7(c) for c in board]
    _eval = eval7.evaluate   # Local reference avoids attribute lookups

    wins = 0.0
    _sample = random.sample   # Local reference avoids repeated global dict lookups
    _choice = random.choice

    for _ in range(n_sims):

        # ----- CASE 1: One opponent card is known; sample the other -----
        if second_card_pool is not None:
            second = _choice(second_card_pool)
            opp_hole_strs = [opp_known[0], second]
            e7_opp = [_e7(c) for c in opp_hole_strs]
            pool = [c for c in remaining if c != second]

        # ----- CASE 2: No opponent cards known -----
        else:
            pair = _choice(range_pairs)
            opp_hole_strs = list(opp_known) + [c for c in pair
                                                if c not in opp_known][:cards_needed_opp]
            e7_opp = [_e7(c) for c in opp_hole_strs]
            pool = [c for c in remaining if c not in opp_hole_strs]

        # Complete the board by sampling from the valid pool (Cases 1 and 2).
        sampled = _sample(pool, cards_needed_board)
        e7_sampled = [_e7(c) for c in sampled]

        my  = _eval(e7_hole + e7_board + e7_sampled)
        opp = _eval(e7_opp  + e7_board + e7_sampled)

        if my > opp:
            wins += 1.0
        elif my == opp:
            wins += 0.5

    # Guard against divide-by-zero if n_sims was 0 (should never occur normally).
    return wins / n_sims if n_sims > 0 else 0.5


def get_equity(state: PokerState,
               n_sims: int = MC_SIMS,
               opp_range_fraction: float = RANGE_LOOSE) -> float:
    """
    Top-level equity dispatcher: routes to the appropriate equity function
    depending on the current street.

    - Pre-flop: use the fast lookup table (preflop_equity).
    - All other streets: use Monte Carlo simulation (monte_carlo_equity).

    Args:
        state:              Current game state (provides hole, board, opp cards).
        n_sims:             Number of MC iterations (ignored pre-flop).
        opp_range_fraction: Opponent range constraint passed to MC.

    Returns:
        Float in [0.0, 1.0] — our estimated equity for this hand/board.
    """
    hole  = state.my_hand
    board = list(state.board)              if state.board              else []
    opp   = list(state.opp_revealed_cards) if state.opp_revealed_cards else []

    # Cannot estimate equity without hole cards.
    if not hole or len(hole) < 2:
        return 0.5

    if state.street == 'pre-flop':
        return preflop_equity(hole)

    return monte_carlo_equity(hole, board, opp, n_sims, opp_range_fraction)


# =============================================================================
# SECTION 7: BOARD TEXTURE ANALYSIS
#
# Board texture describes how "dangerous" or "coordinated" the community cards
# are. A "wet" board (many flush/straight draws) requires different bet sizing
# and bluffing strategy than a "dry" board (uncoordinated, no obvious draws).
# =============================================================================

class BoardTexture:
    """
    Analyzes the community cards and produces a suite of boolean flags and a
    continuous 'wetness' score for strategic use.

    Attributes:
        flush_draw   (bool):  Three or more cards share a suit → flush is possible.
        straight_draw(bool):  Three board cards span within 4 ranks → straight draw possible.
        paired       (bool):  At least two board cards share the same rank.
        trips_on_board(bool): Three board cards share the same rank.
        high_board   (bool):  Average board rank is above 10 (face-card heavy).
        wetness     (float):  Aggregate danger score in [0.0, ~1.0].
        danger      (float):  Alias for wetness (provided for semantic clarity).
    """

    def __init__(self, board: list):
        """
        Compute all texture flags and the wetness score from the board cards.

        Args:
            board: List of card strings on the community board (0–5 cards).
        """
        self.board = board
        ranks = [card_rank(c) for c in board]
        suits = [card_suit(c) for c in board]

        # Flush draw: if any suit appears 3+ times, a flush is possible.
        self.flush_draw = max(suits.count(s) for s in SUITS) >= 3 if board else False

        # Straight draw: delegated to the helper below.
        self.straight_draw = self._has_straight_draw(ranks)

        # Paired board: if any rank repeats (but not all cards are unique).
        self.paired = len(ranks) != len(set(ranks)) and len(board) >= 2

        # Trips on board: a full three-of-a-kind exists on the community cards.
        self.trips_on_board = any(ranks.count(r) >= 3 for r in set(ranks))

        # High board: the average rank is above 10 (mostly face cards).
        self.high_board = (sum(ranks) / max(len(ranks), 1)) > 10 if board else False

        # Wetness is a weighted sum of the above flags, calibrated so that a
        # board with both a flush draw and straight draw maxes out near 0.75+.
        # Weights reflect the relative strategic importance of each factor:
        #   Flush draws (0.40) are slightly more dangerous than straight draws (0.35)
        #   because they are harder to spot and block.
        self.wetness = (
            0.40 * int(self.flush_draw)
          + 0.35 * int(self.straight_draw)
          + 0.15 * int(self.paired)
          + 0.10 * int(self.high_board)
        )

    @staticmethod
    def _has_straight_draw(ranks: list) -> bool:
        """
        Detect whether three or more board cards fall within a 5-card window,
        meaning at least one straight draw exists.

        Algorithm:
            1. Deduplicate and sort the ranks.
            2. Check every window of 3 consecutive unique ranks.
               If the highest minus the lowest ≤ 4, they can all be part of
               the same 5-card straight span.
            3. Also check for a "wheel" straight draw (A-2-3-4-5) by treating
               the Ace as rank 1.

        Args:
            ranks: List of integer rank values from the board cards.

        Returns:
            True if a straight draw exists, False otherwise.
        """
        if len(ranks) < 3:
            return False   # Need at least 3 cards to form a meaningful draw

        uniq = sorted(set(ranks))

        # Check every 3-card window in sorted unique ranks.
        for i in range(len(uniq) - 2):
            if uniq[i + 2] - uniq[i] <= 4:
                return True

        # Special case: Ace can play low (as 1) in a wheel straight A-2-3-4-5.
        if 14 in uniq:
            low_uniq = sorted({1 if r == 14 else r for r in uniq})
            for i in range(len(low_uniq) - 2):
                if low_uniq[i + 2] - low_uniq[i] <= 4:
                    return True

        return False

    def __str__(self):
        """
        Human-readable summary of the board texture, useful for debugging.
        Example output: 'Board(FD|SD, wet=0.75)'
        """
        parts = []
        if self.flush_draw:     parts.append('FD')
        if self.straight_draw:  parts.append('SD')
        if self.paired:         parts.append('Paired')
        if self.high_board:     parts.append('High')
        return f'Board({"|".join(parts) or "Dry"}, wet={self.wetness:.2f})'


def board_texture(board: list) -> BoardTexture:
    """
    Convenience factory function: instantiate a BoardTexture from a list of cards.

    Args:
        board: Community card strings.

    Returns:
        A fully initialized BoardTexture instance.
    """
    return BoardTexture(board)


# =============================================================================
# SECTION 8: STACK-TO-POT RATIO (SPR) UTILITIES
#
# SPR = effective stack size / pot size
#
# SPR tells us how many "pot-sized bets" are left behind. Low SPR means the
# money is already committed and we should rarely fold. High SPR means there
# is plenty of room to maneuver — draws become more valuable and bluffs can
# yield bigger folds.
# =============================================================================

def compute_spr(state: PokerState) -> float:
    """
    Compute the Stack-to-Pot Ratio using the smaller of the two players'
    remaining stacks (the "effective stack").

    A high SPR (>8) → deep-stacked, implied odds matter.
    A low SPR (<3)  → pot-committed, lean toward jam/fold.

    Args:
        state: Current game state with chip counts and pot size.

    Returns:
        Float representing SPR. Returns 20.0 if the pot is zero (pre-bet).
    """
    eff_stack = min(state.my_chips, getattr(state, 'opp_chips', state.my_chips))
    return eff_stack / state.pot if state.pot > 0 else 20.0


def spr_adjustment(spr: float, equity: float) -> float:
    """
    Nudge our effective equity estimate up or down based on the SPR.

    Low SPR (committed pot):
        - If we are already ahead (equity ≥ 55%), push further toward commit (+0.05).
        - If we are behind (equity < 40%), push toward fold (-0.05).
        No adjustments in the middle: we're committed anyway.

    High SPR (deep stacks):
        - Draws in the 30–50% equity range gain implied odds value (+0.04).
        - Monster hands lose a little EV compared to shallow because opponents
          can fold more often before committing all chips (-0.03).

    Args:
        spr:    Stack-to-Pot Ratio (from compute_spr).
        equity: Raw Monte Carlo or table equity for this hand.

    Returns:
        Float adjustment to add to equity (can be negative).
    """
    adj = 0.0

    if spr < SPR_LOW:
        if equity >= 0.56:
            adj += 0.04    # Already ahead; push toward commitment
        elif equity < 0.42:
            adj -= 0.04    # Behind; discourage calling off chips

    elif spr > SPR_HIGH:
        if 0.30 <= equity < 0.54:
            adj += 0.04    # Drawing hands gain implied-odds value
        elif equity > 0.66:
            adj -= 0.03    # Opponents can fold before getting it all in vs. monsters

    else:
        if 0.45 <= equity < 0.56:
            adj += 0.02    # Moderate equity hands gain a small boost in mid SPR

    return adj


# =============================================================================
# SECTION 9: OPPONENT MODEL
#
# The OpponentModel class tracks statistics about the opponent across the
# entire session. It is persistent — it is created once in Player.__init__()
# and updated after every hand.
#
# Key statistics tracked:
#   - Pre-flop aggression rate (VPIP: Voluntarily Put In Pot)
#   - Post-flop bet frequency per street (flop/turn/river)
#   - Fold rate (did they fold before showdown?)
#   - Auction results (who won, did winner bet immediately after?)
#
# The model classifies the opponent into archetypes:
#   Tight/Loose (range width) × Passive/Aggressive (betting frequency)
# These archetypes feed into AdaptiveStrategy to adjust thresholds.
# =============================================================================

class OpponentModel:
    """
    Persistent tracker for opponent behavior statistics across a full session.

    All "record_*" methods are called from Player.on_hand_end() after each hand
    completes. Property accessors are called by AdaptiveStrategy to compute
    adjusted thresholds.
    """

    def __init__(self):
        # ---- Long-term global counters (never reset, session-wide) ----

        self.hands_seen     = 0   # Total hands played this session
        self.pf_total       = 0   # Hands where we observed pre-flop action

        # Per-street tracking: {street: count of times we observed the street}
        self.postflop_opps = {'flop': 0, 'turn': 0, 'river': 0}

        self.showdowns   = 0   # Hands that reached the river

        # Auction-specific counters.
        self.auction_rounds    = 0   # Total auction phases seen
        self.our_bids          = deque(maxlen=200)  # Rolling deque of bid values

        # ---- Post-auction behavior tracking ----
        # Tracks how often the opponent bets the flop immediately after winning
        # the auction — a signal of whether they use the revealed card actively.
        self.opp_won_auction_count      = 0   # Times opp won the auction

        # ---- Exponential Moving Average (EMA) State ----
        # Alpha dictates the decay rate. Equivalent to a ~100 hand window.
        self.ALPHA = 0.04   # ~4% weight to the most recent hand, ~96% to the existing average
        
        # Initialize EMA trackers as None so we can seed them with the first observation
        self.ema_vpip = None
        self.ema_fold_rate = None
        self.ema_opp_auct_bet_rate = None
        self.ema_postflop_agg = None
        self.ema_won_auction = None

    def record_preflop(self, was_aggressive: bool):
        """
        Log whether the opponent played aggressively pre-flop (raised, 3-bet,
        or made any bet beyond completing the blind).

        Args:
            was_aggressive: True if opponent voluntarily put in chips pre-flop.
        """
        self.pf_total     += 1
        
        # Update EMA
        val = float(was_aggressive)
        if self.ema_vpip is None:
            self.ema_vpip = val
        else:
            self.ema_vpip = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_vpip

    def record_street_bet(self, street: str, opp_bet: bool):
        """
        Record whether the opponent made a bet or raise on a given post-flop street.

        Called from Player.on_hand_end() for each street that was actually played.

        Args:
            street:  One of 'flop', 'turn', 'river'.
            opp_bet: True if the opponent bet/raised on that street.
        """
        if street not in self.postflop_opps:
            return   # Ignore unknown streets (safety guard)

        self.postflop_opps[street] += 1

        # Update EMA
        val = float(opp_bet)
        if self.ema_postflop_agg is None:
            self.ema_postflop_agg = val
        else:
            self.ema_postflop_agg = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_postflop_agg

    def record_hand_end(self, payoff: int, final_street: str, opp_revealed: list):
        """
        Update session-wide counters at the conclusion of each hand.

        Determines whether the opponent folded (we won before showdown),
        whether the hand reached showdown, and whether the opponent's cards
        were revealed (meaning we won the auction this hand).

        Args:
            payoff:       Our chip gain/loss for this hand.
            final_street: The last street played ('pre-flop', 'flop', ..., 'river').
            opp_revealed: List of cards the opponent showed (empty if none).
        """
        self.hands_seen += 1

        # If we won chips and the hand didn't reach the river, the opponent folded.
        won_early = payoff > 0 and final_street != 'river'

        if final_street == 'river':
            self.showdowns += 1

        # If the opponent's card was revealed, we must have won the auction.
        auc = float(opp_revealed != [])  # 1.0 if we won the auction, 0.0 if not
        if self.ema_won_auction is None:
            self.ema_won_auction = auc
        else:
            self.ema_won_auction = self.ALPHA * auc + (1.0 - self.ALPHA) * self.ema_won_auction

        # Update EMA
        val = float(won_early)
        if self.ema_fold_rate is None:
            self.ema_fold_rate = val
        else:
            self.ema_fold_rate = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_fold_rate

    def record_auction(self, our_bid: int):
        """
        Record the bid our bot placed in the auction phase.

        Bids are stored in a capped list to allow analysis of bidding patterns
        (though this is currently informational only — bid_multiplier() uses
        win rate, not bid amounts).

        Args:
            our_bid: The chip amount we bid.
        """
        self.auction_rounds += 1
        self.our_bids.append(our_bid)

        # deque(maxlen=200) automatically discards the oldest entry.

    def record_opp_auction_result(self, opp_won: bool, opp_bet_after: bool):
        """
        Record whether the opponent won the auction and, if so, whether they
        immediately bet on the subsequent flop.

        Tracking post-auction bet behavior lets us infer whether the opponent
        uses the revealed card information aggressively (bluffing/value betting)
        or passively (trapping/slow-playing).

        Args:
            opp_won:      True if the opponent won the auction (they saw a card).
            opp_bet_after: True if the opponent bet on the flop after winning.
        """
        if opp_won:
            self.opp_won_auction_count += 1

            # Update EMA
            val = float(opp_bet_after)
            if self.ema_opp_auct_bet_rate is None:
                self.ema_opp_auct_bet_rate = val
            else:
                self.ema_opp_auct_bet_rate = self.ALPHA * val + (1.0 - self.ALPHA) * self.ema_opp_auct_bet_rate

    # ---- Computed rate properties ----

    @property
    def our_auction_win_rate(self) -> float:
        """
        Fraction of auction phases where our bot won and received a card reveal.
        Starts at 0.4 (bet more aggressively) before any auctions are seen.
        """
        return self.ema_won_auction if self.auction_rounds > 8 else 0.4

    @property
    def opp_post_auction_bet_rate(self) -> float:
        """
        How often the opponent bets the flop immediately after winning an auction.

        Uses the recent ema if at least 16 data points exist;
        otherwise defaults to 0.6 as a neutral assumption (slightly aggressive) until we have enough data.

        High rate → opponent uses auction information aggressively.
        Low rate  → opponent may be slow-playing or the auction card didn't help.
        """
        if self.opp_won_auction_count < 16 or self.ema_opp_auct_bet_rate is None:
            return 0.6 # Default to slightly aggressive until we have data
        return self.ema_opp_auct_bet_rate

    @property
    def recent_vpip(self) -> float:
        """
        Recent pre-flop looseness using the exponential moving average.
        Defaults to 0.6 (loose) until we have enough data to trust the EMA.
        """
        if self.pf_total < 16 or self.ema_vpip is None:
            return 0.6  # Default to loose until we have data
        return self.ema_vpip

    @property
    def recent_fold_rate(self) -> float:
        """
        Recent fold rate using only the last _WINDOW hands.
        Falls back to session-wide fold_rate if insufficient data.
        """
        if self.hands_seen < 16 or self.ema_fold_rate is None:
            return 0.16 # Default to a moderate fold rate until we have data
        return self.ema_fold_rate
    
    @property
    def postflop_aggression(self) -> float:
        """
        Overall post-flop aggression using exponential decay.
        High = aggressive, Low = passive/checking.
        """
        to = sum(self.postflop_opps.values())
        
        # Fall back to the raw session average if we have less than 16 post-flop observations
        if to < 16 or self.ema_postflop_agg is None:
            return 0.4 # Default to a moderate aggression rate until we have data
        return self.ema_postflop_agg

    # ---- Archetype classifiers ----

    def is_tight(self)      -> bool: return self.recent_vpip < 0.30           # Plays <30% of hands
    def is_loose(self)      -> bool: return self.recent_vpip > 0.70           # Plays >60% of hands
    def is_aggressive(self) -> bool: return self.postflop_aggression > 0.50   # Bets >50% of streets
    def is_passive(self)    -> bool: return self.postflop_aggression < 0.30   # Bets <30% of streets
    def is_folder(self)     -> bool: return self.recent_fold_rate > 0.30      # Folds >35% pre-showdown
    def sufficient_data(self) -> bool: return self.hands_seen >= MODEL_MIN_HANDS  # Trust the model?

    def inferred_range_fraction(self) -> float:
        """
        Translate the opponent's observed play style into a range fraction
        used to constrain Monte Carlo opponent hand sampling.

        Returns:
            RANGE_TIGHT  (0.30) if opponent plays few hands (tight).
            RANGE_LOOSE  (0.96) if opponent plays many hands (loose) OR
                                 if we don't yet have enough data to trust the model.
            RANGE_MEDIUM (0.50) for everything in between.
        """
        if not self.sufficient_data():
            return RANGE_LOOSE   # Default to no constraint until we have data

        if self.is_tight():
            return RANGE_TIGHT
        if self.is_loose():
            return RANGE_LOOSE
        return RANGE_MEDIUM

    def __str__(self):
        """Debug-friendly string showing key model statistics."""
        return (f'Hands={self.hands_seen}  VPIP={self.ema_vpip:.2f}  '
                f'PF_agg={self.postflop_aggression:.2f}  '
                f'Fold%={self.fold_rate:.2f}  '
                f'AuctWin%={self.our_auction_win_rate:.2f}  '
                f'OppPostAuctBet%={self.opp_post_auction_bet_rate:.2f}')


# =============================================================================
# SECTION 10: ADAPTIVE STRATEGY ENGINE
#
# AdaptiveStrategy sits on top of the OpponentModel and converts behavioral
# statistics into concrete threshold values that govern our actions.
#
# The five thresholds it produces:
#   fold_t:  Fold when equity is below this (facing a bet)
#   call_t:  Call when equity is at least this (facing a bet)
#   value_t: Value bet/raise when equity is at least this
#   bluff_f: Probability of bluffing with a weak hand (random check)
#   small_f: Fraction of pot for thin/merged bets
#   big_f:   Fraction of pot for strong value bets
#
# These are initialized to global constants and then adjusted by:
#   1. Opponent archetype (tight/loose/passive/aggressive)
#   2. Table position (Big Blind / Small Blind)
#   3. Post-auction signals (did they use their auction advantage?)
# =============================================================================

class AdaptiveStrategy:
    """
    Dynamic strategy layer that adjusts decision thresholds based on opponent
    tendencies observed by OpponentModel.
    """

    def __init__(self, model: OpponentModel):
        """
        Args:
            model: The shared OpponentModel instance (same object across the session).
        """
        self.model = model

    def thresholds(self, is_bb: bool = False,
                   opp_won_auction_and_bet: bool = False) -> tuple:
        """
        Compute the six threshold values for the current action decision.

        Process:
            1. Start with global baseline constants.
            2. If insufficient opponent data → apply only positional adjustments.
            3. Otherwise → apply opponent-archetype adjustments first.
            4. Then apply positional modifiers.
            5. Then apply post-auction-aggression modifiers if relevant.

        Args:
            is_bb:                  True if we are the Big Blind this hand.
            opp_won_auction_and_bet: True if opponent won auction AND bet flop.

        Returns:
            Tuple: (fold_t, call_t, value_t, bluff_f, small_f, big_f)
        """
        m = self.model

        # Initialize all thresholds from global constants.
        fold_t  = FOLD_THRESHOLD
        call_t  = CALL_THRESHOLD
        value_t = VALUE_THRESHOLD
        bluff_f = BLUFF_FREQ
        small_f = BET_SMALL_FRAC
        big_f   = BET_BIG_FRAC

        # --- Insufficient data: only apply positional adjustments ---
        if not m.sufficient_data():
            fold_t, call_t, bluff_f = self._apply_position(fold_t, call_t, bluff_f, is_bb)
            if opp_won_auction_and_bet:
                fold_t, call_t = self._apply_opp_auction_aggression(fold_t, call_t, m)
            return fold_t, call_t, value_t, bluff_f, small_f, big_f

        # --- Continuous archetype blending ---
        # Compute continuous tightness/aggression axes in [0, 1] using sigmoid
        # smoothing. This replaces hard binary cutoffs with smooth transitions.
        #   tightness: 0 = tight, 1 = loose
        #   aggression: 0 = passive, 1 = aggressive
        vpip = m.recent_vpip
        agg  = m.postflop_aggression

        def _sigmoid(x, mid, steepness=12.0):
            return 1.0 / (1.0 + math.exp(-steepness * (x - mid)))

        tightness  = _sigmoid(vpip, 0.375)   # midpoint between tight<0.25 and loose>0.50
        aggression = _sigmoid(agg,  0.35)    # midpoint between passive<0.25 and aggro>0.45

        # Bilinear blend weights for the four archetypes.
        w_tp = (1.0 - tightness) * (1.0 - aggression)   # tight-passive
        w_ta = (1.0 - tightness) * aggression            # tight-aggressive
        w_lp = tightness * (1.0 - aggression)            # loose-passive
        w_la = tightness * aggression                     # loose-aggressive

        # Target threshold values for each archetype (match original discrete values).
        #               (fold_t,                          call_t,                          value_t,                          bluff_f,                         small_f, big_f)
        tp = (FOLD_THRESHOLD,                   CALL_THRESHOLD,                  min(VALUE_THRESHOLD + 0.04, 0.75), min(BLUFF_FREQ * 1.8, 0.40),       0.40,    0.60)
        ta = (max(FOLD_THRESHOLD - 0.06, 0.20), max(CALL_THRESHOLD - 0.06, 0.35), max(VALUE_THRESHOLD - 0.04, 0.58), max(BLUFF_FREQ - 0.06, 0.08),     BET_SMALL_FRAC, BET_BIG_FRAC)
        lp = (FOLD_THRESHOLD,                   max(CALL_THRESHOLD - 0.05, 0.35), max(VALUE_THRESHOLD - 0.10, 0.50), max(BLUFF_FREQ * 0.20, 0.03),     0.65,    1.00)
        la = (min(FOLD_THRESHOLD + 0.06, 0.40), min(CALL_THRESHOLD + 0.06, 0.55), min(VALUE_THRESHOLD + 0.05, 0.75), max(BLUFF_FREQ * 0.30, 0.05),     BET_SMALL_FRAC, BET_BIG_FRAC)

        # Weighted blend of all six thresholds.
        fold_t  = w_tp * tp[0] + w_ta * ta[0] + w_lp * lp[0] + w_la * la[0]
        call_t  = w_tp * tp[1] + w_ta * ta[1] + w_lp * lp[1] + w_la * la[1]
        value_t = w_tp * tp[2] + w_ta * ta[2] + w_lp * lp[2] + w_la * la[2]
        bluff_f = w_tp * tp[3] + w_ta * ta[3] + w_lp * lp[3] + w_la * la[3]
        small_f = w_tp * tp[4] + w_ta * ta[4] + w_lp * lp[4] + w_la * la[4]
        big_f   = w_tp * tp[5] + w_ta * ta[5] + w_lp * lp[5] + w_la * la[5]

        # Folder bonus: additive on top of blended result (same condition as before,
        # but using continuous weights to avoid triggering vs. LAG).
        if m.is_folder() and not (tightness > 0.5 and aggression > 0.5):
            bluff_f = min(bluff_f * 1.3, 0.40)

        # --- Apply positional modifiers ---
        fold_t, call_t, bluff_f = self._apply_position(fold_t, call_t, bluff_f, is_bb)

        # --- Apply post-auction aggression modifiers if relevant ---
        if opp_won_auction_and_bet:
            fold_t, call_t = self._apply_opp_auction_aggression(fold_t, call_t, m)

        return fold_t, call_t, value_t, bluff_f, small_f, big_f

    @staticmethod
    def _apply_position(fold_t: float, call_t: float,
                        bluff_f: float, is_bb: bool) -> tuple:
        """
        Adjust thresholds based on positional advantage or disadvantage.

        Big Blind (Out of Position post-flop):
            - Must fold more often when facing bets (no information advantage).
            - Tighten the call requirement.
            - Reduce bluff frequency (bluffs fail more OOP).

        Small Blind (In Position post-flop):
            - Can float wider and bluff with more success.
            - Slightly lower fold threshold.
            - Slightly increase bluff frequency.

        Args:
            fold_t, call_t, bluff_f: Current threshold values.
            is_bb: True if we are the Big Blind.

        Returns:
            Adjusted (fold_t, call_t, bluff_f) tuple.
        """
        if is_bb:
            # Out of position: tighten ranges
            fold_t  = min(fold_t  + 0.03, 0.40)
            call_t  = min(call_t  + 0.02, 0.55)
            bluff_f = max(bluff_f * POSITION_OOP_BLUFF_SCALE, 0.05)
        else:
            # In position: loosen up slightly, bluff more
            fold_t  = max(fold_t  - 0.02, 0.18)
            bluff_f = min(bluff_f * POSITION_IP_BLUFF_SCALE, 0.45)

        return fold_t, call_t, bluff_f

    def _apply_opp_auction_aggression(self, fold_t: float,
                                       call_t: float,
                                       m: 'OpponentModel') -> tuple:
        """
        Adjust fold/call thresholds based on how often the opponent bets after
        winning the auction. Called only when opp_won_auction_and_bet is True.

        If the opponent frequently bets after winning the auction:
            → They are likely betting with actual strength → tighten our ranges.
        If they rarely bet after winning:
            → They may be trapping or bluffing infrequently → allow lighter calls.
        If we don't have enough data yet (< 5 observations):
            → Apply a conservative, diluted tightening as a precaution.

        Args:
            fold_t, call_t: Current threshold values.
            m: The OpponentModel for statistics.

        Returns:
            Adjusted (fold_t, call_t) tuple.
        """
        # Conservative default: not enough data to characterize behavior.
        if m.opp_won_auction_count < 5:
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP * 0.5, 0.45)
            return fold_t, call_t

        rate = m.opp_post_auction_bet_rate

        if rate >= OPP_AUCT_AGG_HIGH:
            # Opponent almost always bets after winning → their hand is real.
            # Raise both fold and call thresholds to play tighter.
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP, 0.48)
            call_t = min(call_t + 0.04, 0.58)

        elif rate <= OPP_AUCT_AGG_LOW:
            # Opponent rarely bets after winning → they may be slow-playing
            # or the card didn't help much. Allow lighter calls.
            call_t = max(call_t - OPP_AUCT_CALL_DIP, 0.33)

        return fold_t, call_t

    def bid_multiplier(self) -> float:
        """
        Scale our raw computed auction bid based on our recent auction win rate.

        If we are losing too many auctions (win rate < 25%), bid higher to
        compete more aggressively and gain more information.

        If we are winning almost all auctions (win rate > 75%), bid lower
        to avoid overpaying for information we tend to get anyway.

        Returns:
            Float multiplier applied to the raw computed bid amount.
        """
        m = self.model

        # Not enough auction history to adjust bidding behavior.
        if m.auction_rounds < 15:
            return 1.0

        wr = m.our_auction_win_rate

        if wr < 0.30:   return 1.60   # Losing lots → bid 60% more aggressively
        elif wr < 0.40: return 1.25   # Slightly low win rate → bid 25% more
        elif wr > 0.8: return 0.70   # Winning most → back off, save chips
        elif wr > 0.6: return 0.90   # Winning plenty → slight pullback
        return 1.0                    # Balanced win rate → no adjustment

    def describe(self) -> str:
        """
        Return a human-readable description of the opponent's inferred archetype.
        Useful for logging and debugging.

        Returns:
            String like 'Tight-Aggressive' or 'Loose-Passive-(Folder)'.
        """
        m = self.model
        if not m.sufficient_data():
            return 'Insufficient data — using defaults'

        parts = []
        if m.is_tight():        parts.append('Tight')
        elif m.is_loose():      parts.append('Loose')
        else:                   parts.append('Normal')

        if m.is_passive():      parts.append('Passive')
        elif m.is_aggressive(): parts.append('Aggressive')
        else:                   parts.append('Balanced')

        if m.is_folder():       parts.append('(Folder)')

        return '-'.join(parts)


# =============================================================================
# SECTION 11: STREET-AWARE BET SIZING
#
# Different streets warrant different bet sizes. Early streets (flop) use
# smaller sizes to build pots gently; later streets (river) use larger sizes
# for maximum extraction. The sizing also shifts based on board wetness:
# bet smaller on wet boards (to avoid charging draws less than pot odds isn't
# the intent here — rather, we size down to avoid over-betting into bad equity).
# =============================================================================

# Base (thin, strong) bet size fractions per street.
# Thin sizing = used for marginal value hands or bluffs.
# Strong sizing = used for premium value hands.
_STREET_SIZING = {
    'pre-flop': (0.4, 0.6),   # Half-pot to three-quarter pot pre-flop
    'flop'    : (0.40, 0.55),   # Slightly under half-pot on the flop
    'turn'    : (0.55, 0.70),   # Growing as pot builds
    'river'   : (0.65, 0.90),   # Large bets to extract maximum value at showdown
}


def street_sizing(street: str, texture: BoardTexture, strong: bool) -> float:
    """
    Determine the appropriate bet sizing fraction for the current street and
    board texture.

    Logic:
        1. Look up (thin, big) fractions for the current street.
        2. Select thin or big based on whether we have a strong hand.
        3. Apply a wetness adjustment: wet boards → size down slightly;
           dry boards → size up slightly. This reflects that on wet boards,
           opponent draws make large bets more likely to get called by equity
           share, and we want to keep the pot manageable.

    Args:
        street:  Current street name.
        texture: Board texture object with wetness score.
        strong:  True if we have a value hand, False for thin/bluff sizing.

    Returns:
        Float fraction of the pot to bet (e.g., 0.65 = 65% pot bet).
    """
    thin_f, big_f = _STREET_SIZING.get(street, (BET_SMALL_FRAC, BET_BIG_FRAC))
    base = big_f if strong else thin_f

    # Wet board adjustment: scale bet size inversely with wetness.
    # wetness=0.5 → no change; wetness=1.0 → -0.10; wetness=0.0 → +0.10
    wet_adj = (texture.wetness - 0.5) * (-0.20)

    return max(0.30, min(1.20, base + wet_adj))


def is_drawing_hand(hole: list, board: list, texture: BoardTexture) -> bool:
    """
    Detect whether our hand is primarily a draw (flush draw or open-ended
    straight draw) rather than a made hand.

    Checks:
      1. Flush draw: at least 1 hole card shares a suit with enough board cards
         to make 4-to-a-flush total.
      2. Straight draw: at least 4 unique ranks within a 5-rank window, with at
         least 1 of our hole cards contributing.

    Args:
        hole:    Our 2 hole cards.
        board:   Community cards (3+).
        texture: Pre-computed BoardTexture.

    Returns:
        True if we appear to be on a significant draw.
    """
    if len(board) < 3:
        return False

    # --- Flush draw check ---
    hole_suits = [card_suit(c) for c in hole]
    board_suits = [card_suit(c) for c in board]
    for s in SUITS:
        hole_count = hole_suits.count(s)
        board_count = board_suits.count(s)
        if hole_count >= 1 and hole_count + board_count >= 4:
            return True

    # --- Straight draw check ---
    all_ranks = sorted(set(card_rank(c) for c in hole + board))
    hole_rank_set = set(card_rank(c) for c in hole)

    # Check every 5-rank window for 4+ ranks with at least 1 hole card.
    for base in range(2, 11):
        window = set(range(base, base + 5))
        in_window = [r for r in all_ranks if r in window]
        if len(in_window) >= 4 and any(r in window for r in hole_rank_set):
            return True

    # Wheel check (A-2-3-4-5): treat Ace as 1.
    wheel = {14, 2, 3, 4, 5}
    wheel_ranks = [r for r in all_ranks if r in wheel]
    if len(wheel_ranks) >= 4 and any(r in wheel for r in hole_rank_set):
        return True

    return False


# =============================================================================
# SECTION 12: AUCTION CARD EXPLOITATION
#
# When we win the auction, we learn one of the opponent's hole cards.
# This section defines how we adjust our behavior based on the rank of
# that revealed card. High card → opponent has partial strength; treat
# cautiously. Low card → opponent's hand is weaker; exploit accordingly.
# =============================================================================

def revealed_card_adjustment(revealed_card: str, board: list, auctions_won: int = 0) -> tuple:
    """
    Compute equity, bluff, and sizing adjustments based on the rank of the
    opponent's card that was revealed through winning the auction.

    Three scenarios:

    1. High card (Jack+, rank ≥ 11):
       The opponent holds at least one strong card. We reduce our equity
       estimate, bluff less (harder to fold them), and bet smaller (less
       value in betting into a potentially strong hand).

    2. Low card (7 or below, rank ≤ 7):
       The opponent holds a weak card. We gain equity, bluff more, and
       bet bigger. However, if that low card connects to the board (within
       2 ranks of a board card), we dampen the boost because even low cards
       can be strong in context (e.g., a paired board or straight draws).

    3. Middle card (8–10):
       Neutral — no significant adjustment either direction.

    Args:
        revealed_card: The card string shown to us from the auction win.
        board:         Current community cards.

    Returns:
        Tuple of (equity_delta, bluff_scale, size_scale) where:
            equity_delta: Added to effective equity (can be negative).
            bluff_scale:  Multiply bluff frequency by this.
            size_scale:   Multiply bet size fraction by this.
    """
    rank = card_rank(revealed_card)
    board_ranks = [card_rank(c) for c in board]

    # Check if the low card could connect to the board (straight/pair proximity).
    connects_to_board = any(abs(rank - br) <= 2 for br in board_ranks)

    # Confidence scalar: ramps from 0.0 → 1.0 as auctions_won approaches
    # AUCTION_CONFIDENCE_MIN. Below that threshold, adjustments are dampened.
    confidence = min(1.0, auctions_won / AUCTION_CONFIDENCE_MIN)

    if rank >= REVEALED_HIGH_RANK:
        # Opponent has a Jack, Queen, King, or Ace → tread carefully.
        eq_d  = REVEALED_HIGH_EQ_DELTA    * confidence
        bl_sc = 1.0 + (REVEALED_HIGH_BLUFF_SCALE - 1.0) * confidence
        sz_sc = 1.0 + (REVEALED_HIGH_SIZE_SCALE  - 1.0) * confidence
        return (eq_d, bl_sc, sz_sc)

    elif rank <= REVEALED_LOW_RANK:
        # Opponent has a 7 or below → they are weak, but dampen if it connects.
        dampen = 0.5 if connects_to_board else 1.0
        eq_d  = REVEALED_LOW_EQ_DELTA * dampen * confidence
        bl_sc = 1.0 + (REVEALED_LOW_BLUFF_SCALE - 1.0) * dampen * confidence
        sz_sc = 1.0 + (REVEALED_LOW_SIZE_SCALE  - 1.0) * dampen * confidence
        return (eq_d, bl_sc, sz_sc)

    # Middle rank (8–10): no meaningful signal, return neutral adjustments.
    return (0.0, 1.0, 1.0)


# =============================================================================
# SECTION 13: AUCTION BIDDING ENGINE
#
# The auction mechanic allows players to bid for the right to see one of the
# opponent's hole cards. The winner pays their bid and the information is
# revealed. Our bidding strategy is grounded in Expected Value:
#
#   EV of winning auction   = equity if we see a card (better decisions post-flop)
#   EV of losing auction    = equity if opponent sees their own card (they play better)
#   Information value       = |eq_win - eq_lose| × pot_size
#   Final bid               = information_value × marginal_factor × adaptive_multiplier
#
# Additional constraints:
#   - Bid SMALL_BLIND if our equity is at the extreme ends (>85% or <15%) — information
#     doesn't change our decisions much at these extremes.
#   - Bid more as BB (out of position) because information is more valuable
#     when you act first and have less information.
#   - Cap the bid as a fraction of the pot to avoid giving away too many chips.
# =============================================================================

def dynamic_max_bid_frac(pot: int, effective_stack: int) -> float:
    """
    Compute the maximum fraction of the pot we are willing to bid.

    The cap shrinks as the pot grows (we've already committed chips; the
    marginal value of information decreases). The cap grows slightly if
    stacks are very deep (more implied value from better decisions post-flop).

    The result is clamped to [BID_CAP_MIN, BID_CAP_MAX].

    Args:
        pot:             Current pot size in chips.
        effective_stack: Smaller of the two players' remaining stacks.

    Returns:
        Float fraction of the pot representing the maximum bid.
    """
    pot_fraction = pot / max(STARTING_STACK, 1)

    # As pot grows to 30%+ of starting stack, pot_cap approaches BID_CAP_MIN.
    pot_cap = BID_CAP_MAX - (BID_CAP_MAX - BID_CAP_MIN) * min(pot_fraction / 0.30, 1.0)

    # Deep stack bonus: if effective stack is more than 30% of starting stack,
    # allow a slightly higher bid cap (more post-flop streets to exploit info).
    stack_fraction = effective_stack / max(STARTING_STACK, 1)
    stack_adj = 0.04 * max(0.0, stack_fraction - 0.3)

    return min(BID_CAP_MAX, max(BID_CAP_MIN, pot_cap + stack_adj))


# Tuning constants for the auction equity estimation sub-routines.
AUC_BATCH_TRIALS = 50  # Total MC trials for batched auction equity estimation
AUC_LOSE_FRAC    = 0.35 # Fallback range tightening if no opponent data (35% of range)


def _dynamic_lose_frac(opp_range_fraction: float, strategy: AdaptiveStrategy) -> float:
    """
    Determine how much the opponent tightens their range when they win the auction.

    Formula:
        dynamic_multiplier = 0.25 + 0.42 * agg_rate
        - agg_rate = 0.0 → multiplier = 0.25 → keep 25% of range (very tight)
        - agg_rate = 1.0 → multiplier = 0.67 → keep 67% of range (fairly wide)

    Falls back to 50% of original range fraction before sufficient data exists.

    Args:
        opp_range_fraction: Inferred range width before auction.
        strategy:           AdaptiveStrategy (provides access to OpponentModel).

    Returns:
        Float — estimated range fraction after opponent wins auction.
    """
    model = strategy.model

    base_frac = opp_range_fraction * 0.50

    if model.sufficient_data():
        agg_rate           = model.opp_post_auction_bet_rate
        dynamic_multiplier = 0.25 + (0.42 * agg_rate)
        base_frac          = opp_range_fraction * dynamic_multiplier

    return max(0.10, min(opp_range_fraction, base_frac))


def _batched_auction_equity(hole: list, board: list,
                            excluded: set,
                            opp_range_fraction: float,
                            strategy: AdaptiveStrategy) -> tuple:
    """
    Estimate equity for both auction outcomes (win / lose) in a single
    batched Monte Carlo pass, eliminating redundant setup work.

    For each trial:
      - Sample an opponent hand from their range.
      - Win scenario: reveal one card, run the board out, compare hands.
      - Lose scenario: opponent plays a tighter range, run the board out.

    Args:
        hole:               Our hole cards.
        board:              Current community cards (usually empty pre-flop).
        excluded:           Set of dead cards.
        opp_range_fraction: Opponent's baseline range width.
        strategy:           AdaptiveStrategy (for lose-scenario range tightening).

    Returns:
        Tuple (eq_win, eq_lose): estimated equity if we win / lose the auction.
    """
    remaining = [c for c in FULL_DECK if c not in excluded]
    range_pairs = build_opponent_range(opp_range_fraction, excluded)

    if not range_pairs:
        range_pairs = [(a, b) for a in remaining for b in remaining if a < b]

    # Tighter range for lose scenario.
    tighter_frac = _dynamic_lose_frac(opp_range_fraction, strategy)
    tighter_pairs = build_opponent_range(tighter_frac, excluded)
    if not tighter_pairs:
        tighter_pairs = range_pairs

    cards_needed_board = 5 - len(board)

    # Pre-convert known cards.
    e7_hole = [_e7(c) for c in hole]
    e7_board = [_e7(c) for c in board]
    _eval = eval7.evaluate

    wins_win  = 0.0
    wins_lose = 0.0
    valid_win  = 0
    valid_lose = 0

    _choice = random.choice
    _sample = random.sample

    for _ in range(AUC_BATCH_TRIALS):
        # --- Win scenario: sample opp hand, reveal one card ---
        pair_w = _choice(range_pairs)
        opp_hand_w = list(pair_w)

        pool_w = [c for c in remaining if c not in opp_hand_w]
        if len(pool_w) >= cards_needed_board:
            board_completion_w = _sample(pool_w, cards_needed_board)
            e7_bc_w = [_e7(c) for c in board_completion_w]
            e7_opp_w = [_e7(c) for c in opp_hand_w]

            my_w  = _eval(e7_hole + e7_board + e7_bc_w)
            opp_w = _eval(e7_opp_w + e7_board + e7_bc_w)
            if my_w > opp_w:
                wins_win += 1.0
            elif my_w == opp_w:
                wins_win += 0.5
            valid_win += 1

        # --- Lose scenario: opponent plays tighter range ---
        pair_l = _choice(tighter_pairs)
        opp_hand_l = list(pair_l)

        pool_l = [c for c in remaining if c not in opp_hand_l]
        if len(pool_l) >= cards_needed_board:
            board_completion_l = _sample(pool_l, cards_needed_board)
            e7_bc_l = [_e7(c) for c in board_completion_l]
            e7_opp_l = [_e7(c) for c in opp_hand_l]

            my_l  = _eval(e7_hole + e7_board + e7_bc_l)
            opp_l = _eval(e7_opp_l + e7_board + e7_bc_l)
            if my_l > opp_l:
                wins_lose += 1.0
            elif my_l == opp_l:
                wins_lose += 0.5
            valid_lose += 1

    eq_win  = wins_win  / max(valid_win,  1)
    eq_lose = wins_lose / max(valid_lose, 1)
    return (eq_win, eq_lose)


def compute_bid(state: PokerState,
                strategy: AdaptiveStrategy,
                opp_range_fraction: float = RANGE_LOOSE) -> int:
    """
    Compute the optimal auction bid based on the Expected Value of information.

    Algorithm:
        1. Compute baseline equity without any information advantage.
        2. Short-circuit to bid 0 if equity is extreme (≥84% or ≤16%):
           at these extremes, knowing one card rarely changes our best action.
        3. Estimate eq_win = equity if we see the opponent's card.
        4. Estimate eq_lose = equity if opponent sees their own card.
        5. Information value = |eq_win - eq_lose| × pot (in chips).
        6. Apply marginal_factor = 4 × eq × (1-eq): peaks at 50% equity, with minimum 0.64 cap
           because near-even equities benefit most from information.
        7. Apply adaptive multiplier from AdaptiveStrategy (based on auction
           win rate history).
        8. Multiply by OOP_BID_MULTIPLIER if we are the Big Blind.
        9. Enforce minimum bid and clamp to dynamic pot-fraction cap.

    Args:
        state:              Current game state (provides pot, chips, etc.).
        strategy:           AdaptiveStrategy for bid multiplier lookup.
        opp_range_fraction: Opponent range constraint.

    Returns:
        Integer chip amount to bid (may be 0 to skip the auction).
    """
    hole  = list(state.my_hand)
    board = list(state.board)              if state.board              else []
    opp   = list(state.opp_revealed_cards) if state.opp_revealed_cards else []

    # Step 1: Baseline equity (no auction information).
    equity = monte_carlo_equity(hole, board, opp,
                                n_sims=MC_SIMS_AUCTION,
                                opp_range_fraction=opp_range_fraction)

    # Step 2: Information is irrelevant at the extremes — we know what to do.
    if equity >= 0.84 or equity <= 0.16:
        return SMALL_BLIND

    excluded = set(hole + board + opp)

    # Steps 3 & 4: Estimate equity in both auction outcomes (batched).
    eq_win, eq_lose = _batched_auction_equity(hole, board, excluded,
                                              opp_range_fraction, strategy)

    # Step 5: Raw information value in chips.
    info_value = abs(eq_win - eq_lose) * state.pot

    # Step 6: Marginal factor — peaks at equity=0.50 (maximum uncertainty).
    # 4 × 0.5 × 0.5 = 1.0 (maximum); Cap at 0.64 to prevent overbidding on very strong marginal hands.
    marginal_factor = min(4.0 * equity * (1.0 - equity), 0.64)

    # Step 7: Adaptive bid multiplier from auction win rate history.
    adapt_mult = strategy.bid_multiplier()
    raw_bid    = info_value * marginal_factor * adapt_mult

    # Step 8: Out-of-position premium (BB bids more for information).
    if state.is_bb:
        raw_bid *= OOP_BID_MULTIPLIER

    # Enforce a minimum bid: at least 1 big blind if our auction win rate is
    # poor (< 30%), otherwise 1 small blind. Prevents bidding trivially small amounts.
    min_bid  = BIG_BLIND
    if strategy.model.our_auction_win_rate <= 0.3:
        min_bid += SMALL_BLIND
    elif strategy.model.our_auction_win_rate > 0.7:
        min_bid -= SMALL_BLIND
    
    raw_bid  = max(raw_bid, min_bid)

    # Step 9: Cap the bid as a dynamic fraction of the pot.
    eff_stack    = min(state.my_chips, getattr(state, 'opp_chips', state.my_chips))
    dyn_cap_frac = dynamic_max_bid_frac(state.pot, eff_stack)
    max_bid      = int(state.pot * dyn_cap_frac)
    bid          = int(raw_bid)

    # Final clamp: bid must not exceed our chip stack or the pot cap.
    return min(bid, max_bid, state.my_chips)


# =============================================================================
# SECTION 14: CORE ACTION DECISION ENGINE
#
# decide_action() is the final output stage. It takes equity, adjusted
# thresholds, board texture, and auction context and returns a concrete
# game action: Fold, Check, Call, or Raise.
#
# Decision flow:
#   1. Unpack auction context flags.
#   2. Fetch adaptive thresholds from AdaptiveStrategy.
#   3. Apply auction-loss tightening if opponent won the auction.
#   4. Compute pot odds for calling decisions.
#   5. Adjust equity based on SPR, position, and revealed card.
#   6. Compute effective bluff frequency (texture-weighted).
#   7. If SPR is critically low → jam/fold immediately (skip nuanced logic).
#   8. Route 1 (we can check / we are the aggressor):
#       a. Strong hand → value raise.
#       b. Marginal hand → thin value raise.
#       c. Weak hand → stochastic bluff or check.
#   9. Route 2 (facing a bet from opponent):
#       a. Strong hand → reraise, or call if no raise available.
#       b. Marginal hand (meets pot odds) → call.
#       c. Weak hand → fold.
#  10. Fallback handlers ensure we never return None.
# =============================================================================

def estimate_shove_range(opp_model) -> float:
    """
    Estimate the range fraction an opponent shoves with pre-flop.
    Tighter than their general VPIP — shoving ranges are narrower.
    Falls back to a conservative default before sufficient data.
    """
    if not opp_model.sufficient_data():
        return 0.16   # Default: assume top 15% when no data

    # Shove range is roughly half their standard inferred range,
    # scaled down further if they are passive pre-flop.
    base_frac = opp_model.inferred_range_fraction() * 0.50
    if opp_model.preflop_aggression_rate < 0.30:
        base_frac *= 0.70   # Passive opponents shove even tighter

    return max(0.08, min(0.32, base_frac))

def decide_action(state: PokerState, equity: float,
                  strategy: AdaptiveStrategy,
                  texture: BoardTexture,
                  auction_ctx: dict | None = None):
    """
    Determine the best action to take given the current game state and context.

    This is the core strategic decision function. It synthesizes all analysis
    (equity, thresholds, SPR, texture, position, auction context) into a single
    concrete game action.

    Args:
        state:       Current game state object.
        equity:      Raw equity estimate from get_equity().
        strategy:    AdaptiveStrategy instance providing adjusted thresholds.
        texture:     BoardTexture object for the current board.
        auction_ctx: Optional dict containing auction-related flags:
                       'we_won_auction':       True if we saw the opponent's card.
                       'revealed_card':        The card string we saw (or None).
                       'opp_won_auction':      True if opponent saw their card.
                       'opp_bet_post_auction': True if opp bet flop after winning.
                       'is_bb':                True if we are the Big Blind.

    Returns:
        One of ActionFold, ActionCall, ActionCheck, ActionRaise.
    """
    # --- Step 1: Unpack auction context ---
    ctx = auction_ctx or {}
    we_won_auction       = ctx.get('we_won_auction', False)
    revealed_card        = ctx.get('revealed_card', None)
    opp_won_auction      = ctx.get('opp_won_auction', False)
    opp_bet_post_auction = ctx.get('opp_bet_post_auction', False)
    is_bb                = ctx.get('is_bb', False)
    street_reraise_count = ctx.get('street_reraise_count', 0)

    # Combined flag: opponent won the auction AND immediately bet → likely strong.
    opp_won_and_bet = opp_won_auction and opp_bet_post_auction

    # --- Step 2: Fetch adaptive thresholds ---
    fold_t, call_t, value_t, bluff_f, _sf, _bf = strategy.thresholds(
        is_bb=is_bb, opp_won_auction_and_bet=opp_won_and_bet)
    
    # Reraise pressure adjustment: each opponent reraise signals a stronger hand,
    # so require more equity before betting for value.
    if street_reraise_count > 0:
        value_t = min(0.90, value_t * (1.0 + 0.1 * street_reraise_count))

    # --- Step 3: Tighten ranges if opponent won the auction ---
    # When the opponent has superior information, we should be more conservative.
    # AUCTION_LOSS_K = 0.84 applies a 16% tightening effect.
    if opp_won_auction:
        # Direct multiplication works for bluff_f since it's a probability.
        bluff_f *= AUCTION_LOSS_K

        # Shift equity thresholds upward. Formula: new_t = 1 - K*(1-old_t)
        # This preserves the proportional distance from 1.0, just compressed.
        call_t  = 1.0 - AUCTION_LOSS_K * (1.0 - call_t)
        value_t = 1.0 - AUCTION_LOSS_K * (1.0 - value_t)

        # Ensure fold threshold doesn't accidentally exceed the new call threshold.
        fold_t = min(fold_t, call_t - 0.05)

    # --- Step 4: Basic pot geometry ---
    pot = state.pot
    ctc = state.cost_to_call   # Chips To Call (0 if no bet to face)
    hole = list(state.my_hand)

    # Pot odds: the minimum equity needed to break even on a call.
    pot_odds = ctc / (pot + ctc) if (ctc > 0 and pot + ctc > 0) else 0.0

    min_raise, max_raise = state.raise_bounds

    def clamp(amount: int, jitter: float = 0.04) -> int:
        """
        Clamp a proposed raise amount to the legal raise bounds.
        The engine requires raises to fall within [min_raise, max_raise].
        """
        noise = 1.0 + random.uniform(-jitter, jitter)
        jittered = int(amount * noise)
        return max(min_raise, min(jittered, max_raise))


    # --- Step 5: Effective equity (equity + all adjustments) ---

    # SPR adjustment: committed pots favor made hands; deep stacks favor draws.
    spr = compute_spr(state)
    adj = spr_adjustment(spr, equity)
    eff_eq = max(0.0, min(1.0, equity + adj))

    # Positional adjustment: being in position (SB) gives a small equity bonus;
    # being out of position (BB) gives a penalty.
    street = state.street
    if street != 'pre-flop':
        if is_bb:
            eff_eq = max(0.0, eff_eq + POSITION_OOP_EQ_DELTA)
        else:
            eff_eq = min(1.0, eff_eq + POSITION_IP_EQ_BONUS)
    elif street == 'pre-flop' and is_bb:
        # BB is last to act pre-flop → slight calling range bonus.
        call_t = max(call_t - POSITION_BB_CALL_BONUS, 0.30)
        value_t = value_t * 1.10  # Require stronger hands to 3-bet from the BB pre-flop

    # Revealed card adjustment: modify equity, bluff scale, and size scale
    # based on the specific card revealed from the auction win.
    rev_eq_delta = 0.0
    rev_bluff_sc = 1.0
    rev_size_sc  = 1.0
    if we_won_auction and revealed_card and street != 'pre-flop':
        board = list(state.board) if state.board else []
        rev_eq_delta, rev_bluff_sc, rev_size_sc = revealed_card_adjustment(
            revealed_card, board, auctions_won=strategy.model.ema_won_auction)
        eff_eq = max(0.0, min(1.0, eff_eq + rev_eq_delta))

    # --- Step 5b: Draw detection for sizing/threshold adjustments ---
    board_list = list(state.board) if state.board else []
    is_draw = False
    if street in ('flop', 'turn') and len(board_list) >= 3:
        is_draw = is_drawing_hand(list(state.my_hand), board_list, texture)

    # Draws get implied-odds bonus on flop/turn (lower call threshold).
    if is_draw:
        call_t = max(call_t - 0.05, 0.30)
        # Deep stacks amplify implied odds for draws.
        if spr > SPR_HIGH:
            call_t = max(call_t - 0.03, 0.27)

    # --- Step 6: Effective bluff frequency ---
    # Adjust bluffing frequency for board wetness:
    # Wet boards → bluff less (opponent has many draws and calls more).
    # Dry boards → bluff more (opponent has fewer continuing hands).
    # texture_bluff_scale: at wetness=0.5 → 1.0; at 0 → 1.2; at 1 → 0.8.
    texture_bluff_scale = 1.0 + 0.4 * (0.5 - texture.wetness)
    eff_bluff_f = max(0.0, min(0.50, bluff_f * texture_bluff_scale * rev_bluff_sc))

    # --- Step 7: Compute bet sizing fraction for this action ---
    strong  = eff_eq >= value_t
    sz_frac = street_sizing(street, texture, strong) * rev_size_sc

    # Draw vs. made hand sizing distinction:
    # - Semi-bluffs with draws: use smaller sizing (risk less with incomplete hand).
    # - Made hands on wet boards: use larger sizing (charge opponents drawing against us).
    if is_draw and not strong:
        sz_frac *= 0.80
    elif not is_draw and strong and texture.wetness > 0.35:
        sz_frac *= 1.15

    sz_frac = max(0.25, min(1.40, sz_frac))

    # --- Step 7 (alt): Jam/Fold shortcut for very low SPR situations ---
    # When the effective stack is less than 3× the pot, we are pot-committed.
    # In this case, skip nuanced sizing: jam if ahead, fold/check if behind.
    if spr < SPR_LOW and street != 'pre-flop':
        if eff_eq >= 0.60:
            # We are a slight favorite or better → go all-in / call.
            if eff_eq < 0.64 and state.can_act(ActionCall):
                return ActionCall()
            elif state.can_act(ActionRaise):
                return ActionRaise(max_raise)
            elif state.can_act(ActionCall):
                return ActionCall()
        else:
            # We are behind → get out or check if check is free.
            if state.can_act(ActionCheck):
                return ActionCheck()
            elif state.can_act(ActionFold):
                return ActionFold()

    # =========================================================================
    # ROUTE 1: We can check (no bet to face / we are the aggressor).
    # =========================================================================
    if state.can_act(ActionCheck):

        # --- Check-raise trap ---
        # Against aggressive opponents, occasionally check strong hands to
        # induce a bet, then reraise on the next action (ROUTE 2 handles it).
        if (eff_eq >= value_t
                and street in ('flop', 'turn')
                and strategy.model.sufficient_data()
                and strategy.model.postflop_aggression >= CHECK_RAISE_MIN_AGG
                and random.random() < CHECK_RAISE_FREQ):
            return ActionCheck()

        # --- River-specific polarized logic ---
        if street == 'river':
            if eff_eq >= value_t:
                # Strong hand: polarized large value bet.
                if state.can_act(ActionRaise):
                    river_sz = RIVER_VALUE_SIZE * rev_size_sc
                    return ActionRaise(clamp(pot + int(pot * river_sz)))
                return ActionCheck()
            elif eff_eq < fold_t and random.random() < eff_bluff_f * RIVER_BLUFF_REDUCTION:
                # River bluff: reduced frequency, same large sizing as value (GTO polarized).
                if state.can_act(ActionRaise):
                    river_sz = RIVER_VALUE_SIZE * rev_size_sc
                    return ActionRaise(clamp(pot + int(pot * river_sz)))
                return ActionCheck()
            else:
                # Marginal hands and failed bluff checks: just check.
                return ActionCheck()

        # Strong made hand → fire a full value bet (non-river streets).
        if eff_eq >= value_t:
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * sz_frac)))
            return ActionCheck()

        # Marginal value hand → make a thin bet to build the pot.
        elif eff_eq >= call_t:
            thin_sz = street_sizing(street, texture, False) * rev_size_sc
            thin_sz = max(0.25, min(1.40, thin_sz))
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * thin_sz)))
            return ActionCheck()

        # Weak hand → stochastic bluff or give up with a check.
        else:
            if (street in ('flop', 'turn')
                    and eff_eq < fold_t
                    and random.random() < eff_bluff_f
                    and state.can_act(ActionRaise)):
                bluff_sz = street_sizing(street, texture, False) * rev_size_sc
                bluff_sz = max(0.25, min(1.40, bluff_sz))
                return ActionRaise(clamp(pot + int(pot * bluff_sz)))
            return ActionCheck()

    # =========================================================================
    # ROUTE 2: Facing a bet from the opponent.
    # =========================================================================

    else:
        # --- Pre-flop near-all-in: shove/fold decision ---
        if ctc >= state.my_chips * 0.8 and street == 'pre-flop':
            shove_range = estimate_shove_range(strategy.model)
            eff_eq = monte_carlo_equity(hole, [], [], MC_SIMS_AUCTION, shove_range)
            return ActionCall() if eff_eq > 0.54 else ActionFold()

        # --- River-specific adjustments when facing a bet ---
        if street == 'river':
            # vs passive on river: lower call threshold (exploit value-heavy bets).
            # vs aggressive on river: also lower call threshold (pick off bluffs).
            if strategy.model.sufficient_data() and strategy.model.is_passive():
                call_t_river = max(call_t - 0.04, 0.30)
            elif strategy.model.sufficient_data() and strategy.model.is_aggressive():
                call_t_river = max(call_t - 0.03, 0.32)
            else:
                call_t_river = call_t

            if eff_eq >= value_t:
                if state.can_act(ActionRaise):
                    return ActionRaise(clamp(int((pot + ctc) * 0.81)))
                if state.can_act(ActionCall):
                    return ActionCall()
            elif eff_eq >= max(call_t_river, pot_odds * 1.10):
                if state.can_act(ActionCall):
                    return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()
        else:
            # Non-river streets: original logic.
            if eff_eq >= value_t:
                if state.can_act(ActionRaise):
                    return ActionRaise(clamp(int((pot + ctc) * 0.81)))
                if state.can_act(ActionCall):
                    return ActionCall()
            elif eff_eq >= max(call_t, pot_odds * 1.10):
                if state.can_act(ActionCall):
                    return ActionCall()
            if state.can_act(ActionFold):
                return ActionFold()

    # =========================================================================
    # FALLBACK: Should never reach here under normal circumstances.
    # These guards prevent the bot from crashing if all primary paths fail.
    # =========================================================================
    if state.can_act(ActionCall):   return ActionCall()
    if state.can_act(ActionCheck):  return ActionCheck()
    return ActionFold()


# =============================================================================
# SECTION 15: PLAYER (MAIN BOT CONTROLLER)
#
# The Player class is the integration point between the game engine and the
# internal strategy/model logic. The engine calls:
#   - on_hand_start()  at the beginning of every new hand
#   - on_hand_end()    after a hand finishes
#   - get_move()       whenever the bot must take an action
#
# Player creates OpponentModel and AdaptiveStrategy once at construction,
# so they persist across the entire session and accumulate data.
# =============================================================================

class Player(BaseBot):
    """
    Main bot controller. Manages session-level state (OpponentModel,
    AdaptiveStrategy) and hand-level transient state, routing observations
    and actions through the appropriate subsystems.
    """

    def __init__(self) -> None:
        """
        Initialize session-persistent objects and reset all transient
        hand-level variables.

        OpponentModel and AdaptiveStrategy are created once here and
        live for the entire session. All other attributes are reset
        at the start of each hand by on_hand_start().
        """
        # Session-persistent: these accumulate data across all hands.
        self.opp_model  = OpponentModel()
        self.strategy   = AdaptiveStrategy(self.opp_model)

        self._street_reraise_count  = 0   # Opponent reraises seen on the current street
        self._last_seen_street      = None  # Track street transitions for reset

        # Transient hand-level variables — reset by on_hand_start() each hand.
        self._pf_recorded       = False   # Flag: have we logged pre-flop action?
        self._streets_seen      = set()   # Streets we have observed this hand
        self._opp_bet_streets   = set()   # Streets on which opponent bet/raised
        self._our_auction_bid   = 0       # The bid we placed this hand (0 if none)
        self._is_bb             = False   # Are we the Big Blind this hand?
        self._we_won_auction    = False   # Did we win the auction reveal?
        self._revealed_card     = None    # The specific card revealed to us (or None)
        self._opp_won_auction   = False   # Did the opponent win the auction?
        self._opp_bet_post_auction = False  # Did opp bet the flop after winning?
        self._auction_seen      = False   # Have we processed the auction outcome?

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        """
        Reset all hand-level transient state at the beginning of a new hand.

        Called by the engine before the first action of every hand. Without
        this reset, data from the previous hand would bleed into the new one.

        Args:
            game_info:     Static session metadata (stack sizes, etc.).
            current_state: Initial state of the new hand.
        """
        self._pf_recorded          = False
        self._streets_seen         = set()
        self._opp_bet_streets      = set()
        self._our_auction_bid      = 0
        self._is_bb                = current_state.is_bb   # Determined by seat this hand
        self._we_won_auction       = False
        self._revealed_card        = None
        self._opp_won_auction      = False
        self._opp_bet_post_auction = False
        self._auction_seen         = False
        self._street_reraise_count  = 0
        self._last_seen_street      = None

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """
        Commit all observations from the completed hand to the OpponentModel.

        Called by the engine after a hand finishes (fold, showdown, or all-in
        runout). This is the only place where we write to the persistent model.

        Records:
            - Post-flop street betting behavior (per street observed).
            - Our auction bid (if we placed one).
            - Auction outcome (who won, did winner bet flop).
            - Final hand result (payoff, last street, opponent cards shown).

        Args:
            game_info:     Static session metadata.
            current_state: Final state after the hand concluded.
        """
        # Record per-street betting observations: was each street seen, and
        # did the opponent bet on it?
        for street in ('flop', 'turn', 'river'):
            if street in self._streets_seen:
                self.opp_model.record_street_bet(
                    street, street in self._opp_bet_streets)

        # Record our auction bid if we participated in the auction this hand.
        if self._our_auction_bid > 0:
            self.opp_model.record_auction(self._our_auction_bid)

        # Retrieve any cards the opponent revealed at showdown.
        opp_revealed = (list(current_state.opp_revealed_cards)
                        if current_state.opp_revealed_cards else [])

        # Record auction outcome if an auction occurred this hand.
        if self._auction_seen:
            self.opp_model.record_opp_auction_result(
                opp_won      = self._opp_won_auction,
                opp_bet_after = self._opp_bet_post_auction,
            )

        # Record the final hand outcome: payoff, last street, and showdown cards.
        self.opp_model.record_hand_end(
            payoff       = current_state.payoff,
            final_street = current_state.street,
            opp_revealed = opp_revealed,
        )

    def _observe_preflop(self, state: PokerState):
        """
        Detect pre-flop aggression and log it to the opponent model.

        Called every action on the pre-flop street, but uses _pf_recorded
        to ensure we only log once per hand. If the cost to call is more
        than the standard BB-SB differential, the opponent raised.

        Args:
            state: Current game state.
        """
        if self._pf_recorded or state.street != 'pre-flop':
            return   # Already recorded or wrong street

        # cost_to_call > (BB - SB) means the opponent raised above the initial bet.
        was_aggressive = state.cost_to_call > (BIG_BLIND - SMALL_BLIND)
        self.opp_model.record_preflop(was_aggressive)
        self._pf_recorded = True

    def _observe_postflop(self, state: PokerState):
        """
        Track which post-flop streets we see and whether the opponent bets
        on each street. This data feeds into the post-flop aggression rate.

        A positive cost_to_call on a post-flop street means the opponent bet
        or raised before this action was requested.

        Args:
            state: Current game state.
        """
        s = state.street
        if s not in ('flop', 'turn', 'river'):
            return   # Pre-flop and auction streets are handled separately

        if state.cost_to_call > 0:
            self._opp_bet_streets.add(s)   # Opponent bet on this street
        self._streets_seen.add(s)           # We witnessed this street

    def _observe_auction_outcome(self, state: PokerState):
        """
        Determine who won the auction by examining the game state immediately
        after the auction phase.

        Called the first time we see the flop (which follows the auction).
        Only processes once per hand via the _auction_seen guard.

        Detection logic:
            - If opp_revealed_cards is populated → we won (we see their card).
            - Else if we placed a non-zero bid → opponent won (we bid but got nothing).

        Args:
            state: Current game state (at the start of the flop).
        """
        if self._auction_seen:
            return   # Only process auction outcome once per hand

        self._auction_seen = True
        opp_revealed = (list(state.opp_revealed_cards)
                        if state.opp_revealed_cards else [])

        if opp_revealed:
            # We can see the opponent's card → we won the auction.
            self._we_won_auction = True
            self._revealed_card  = opp_revealed[0]   # Store for use in decide_action()
            self._opp_won_auction = False

        elif self._our_auction_bid > 0:
            # We bid but received no reveal → opponent outbid us.
            self._opp_won_auction = True

    def _observe_opp_post_auction_bet(self, state: PokerState):
        """
        Detect whether the opponent bets on the flop immediately after winning
        the auction. This flag feeds into the post-auction aggression metric
        in OpponentModel.

        Uses a sentinel value ('flop_post_auction_checked') in _streets_seen
        to ensure this check runs only once per hand.

        Args:
            state: Current game state (flop only).
        """
        if (self._opp_won_auction
                and state.street == 'flop'
                and 'flop_post_auction_checked' not in self._streets_seen):

            # Mark as checked so this block doesn't run again this hand.
            self._streets_seen.add('flop_post_auction_checked')

            # If there's a bet facing us on the flop, opponent used their auction info.
            if state.cost_to_call > 0:
                self._opp_bet_post_auction = True

    def _observe_reraises(self, state: PokerState):
        """
        Track how many times the opponent has reraised on the current street.
        Resets the counter whenever the street changes.

        A reraise is detected when cost_to_call exceeds the previous cost_to_call
        on the same street — meaning the opponent raised on top of our last action.
        """
        s = state.street

        # Street changed → reset counter and last-call tracking
        if s != self._last_seen_street:
            self._street_reraise_count = 0
            self._last_ctc             = 0
            self._last_seen_street     = s

        # A reraise: opponent raised again after we had already faced a bet
        if state.cost_to_call > 0 and state.cost_to_call > getattr(self, '_last_ctc', 0):
            self._street_reraise_count += 1

        self._last_ctc = state.cost_to_call


    def get_move(self, game_info: GameInfo, current_state: PokerState) \
            -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        """
        Main decision entry point, called by the game engine on every action.

        Orchestrates the full decision pipeline:
            1. Observe and log pre-flop/post-flop context.
            2. Retrieve opponent range inference.
            3. Handle auction bidding as a special-case street.
            4. On the flop, resolve and cache auction outcomes.
            5. Compute equity via MC simulation (or table lookup pre-flop).
            6. Analyze board texture.
            7. Bundle auction context into a dict for decide_action().
            8. Delegate to decide_action() and return the result.

        Args:
            game_info:     Static session metadata.
            current_state: Current game state requiring an action.

        Returns:
            An action object: ActionFold, ActionCall, ActionCheck,
            ActionRaise, or ActionBid.
        """
        # Step 1: Update pre-flop and post-flop observation trackers.
        self._observe_preflop(current_state)
        self._observe_postflop(current_state)
        self._observe_reraises(current_state)

        # Step 2: Retrieve inferred opponent range from the model.
        # This fraction constrains the MC simulation's opponent hand sampling.
        opp_rf = self.opp_model.inferred_range_fraction()

        # Step 3: Auction street — compute and return a bid immediately.
        # This bypasses all equity/texture logic since bidding is entirely
        # different from normal bet/raise/call/fold decisions.
        if current_state.street == 'auction':
            bid = compute_bid(current_state, self.strategy, opp_rf)
            self._our_auction_bid = bid   # Cache bid for on_hand_end() tracking
            return ActionBid(bid)

        # Step 4: Flop — resolve who won the auction before making decisions.
        # Also check if the opponent immediately bet after winning.
        if current_state.street == 'flop':
            self._observe_auction_outcome(current_state)
            self._observe_opp_post_auction_bet(current_state)

        # Step 5: Compute equity for the current hand/board state.
        equity = get_equity(current_state, MC_SIMS, opp_rf)

        # Step 6: Analyze board texture (wetness, flush draws, etc.).
        board   = list(current_state.board) if current_state.board else []
        texture = board_texture(board)

        # Step 7: Bundle auction context into a clean dict for decide_action().
        auction_ctx = {
            'we_won_auction':       self._we_won_auction,
            'revealed_card':        self._revealed_card,
            'opp_won_auction':      self._opp_won_auction,
            'opp_bet_post_auction': self._opp_bet_post_auction,
            'is_bb':                self._is_bb,
            'street_reraise_count':  self._street_reraise_count,
        }

        # Step 8: Delegate to the core decision engine and return the action.
        return decide_action(current_state, equity, self.strategy, texture, auction_ctx)


# =============================================================================
# SECTION 16: ENTRY POINT
#
# Connects the Player bot to the game server runner when executed directly.
# parse_args() reads command-line arguments (host, port, etc.);
# run_bot() starts the client loop, calling get_move() for each action.
# =============================================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())
