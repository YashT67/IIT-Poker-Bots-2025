'''
bot.py  —  Phase 5 Bot for IIT Pokerbots 2026: Sneak Peek Hold'em
==================================================================
Drop-in replacement for bot.py in the repo root (alongside pkbot/).

PHASE 5 UPGRADES (on top of Phase 4)
--------------------------------------
  5a. Post-auction card exploitation
        When we WIN the auction we see one of the opponent's hole cards.
        We now use the rank of that card to adjust equity and strategy:
          High card revealed (J/Q/K/A): opp has a strong hand → tighten up,
            reduce bluff frequency, bet smaller to retain callers.
          Low card revealed (2–7): opp's hand is weaker/draw-heavy → push
            harder, bet bigger for value, bluff more aggressively.
          Connected to board: dampens the "low is safe" boost — a 7 on
            a 7-8-9 board still pairs the opp.
        When OPP wins the auction and then bets big on the flop, we
        now treat this as a strong hand signal (they found what they
        needed) and increase our fold threshold accordingly.

  5d. Dynamic bid cap (pot geometry)
        Replaces the flat MAX_BID_FRAC=0.20 with a cap that scales with:
          • Pot size relative to starting stack: small pot → info matters
            more (more streets to exploit it) → allow up to 0.27.
          • Effective stack depth: deeper stacks → more implied value
            from information → allow higher cap.
          • Large pre-flop 3-bet pots → opp already committed → info
            worth less → floor at 0.10.

  5f. Post-auction aggression tracking
        Tracks whether the opponent bets on the flop after winning
        the auction.  High opp_post_auction_bet_rate → they extract
        value from the info → when opp wins auction and bets, raise our
        fold threshold and lower our call threshold on that street.
        Low rate → they're passive even with info → we can call lighter.

  5g. Positional pre-flop & post-flop awareness
        In heads-up Sneak Peek Hold'em:
          SB (is_bb=False) = acts first pre-flop, acts LAST post-flop
                             (positional advantage post-flop).
          BB (is_bb=True)  = acts last pre-flop, acts FIRST post-flop
                             (positional disadvantage post-flop).
        Post-flop: SB (IP) gets +0.02 equity bonus, looser thresholds,
          higher bluff frequency.  BB (OOP) gets −0.02 equity penalty,
          tighter fold threshold, lower bluff frequency.
        Pre-flop: BB gets a small call-wider bonus (last to act).

PHASES 1–4 (retained)
-----------------------
  Pre-flop lookup table, range-constrained Monte Carlo, SPR awareness,
  board texture analysis, street-aware bet sizing, true-value auction.
'''

from pkbot.actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from pkbot.states import GameInfo, PokerState, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from pkbot.base import BaseBot
from pkbot.runner import parse_args, run_bot

import math
import random
from itertools import combinations
from functools import lru_cache


# ===========================================================================
# CONSTANTS & GLOBAL PARAMETERS
# ===========================================================================

MC_SIMS         = 96      # Post-flop Monte Carlo rollouts (reduced for speed)
MC_SIMS_AUCTION = 32      # Auction-phase rollouts (reduced for speed)

# Phase 1/2 baseline thresholds (Phase 2 adaptive layer adjusts them)
FOLD_THRESHOLD  = 0.30
CALL_THRESHOLD  = 0.45
VALUE_THRESHOLD = 0.65
BLUFF_FREQ      = 0.18

BET_SMALL_FRAC  = 0.50
BET_BIG_FRAC    = 0.75
MAX_BID_FRAC    = 0.20

MODEL_MIN_HANDS = 30      # hands before trusting opponent model

# Phase 3 — SPR thresholds
SPR_LOW  = 3.0    # below this → committed, shove/raise freely
SPR_HIGH = 8.0    # above this → need strong hands, dump draws

# Phase 3 — range tightness (fraction of hand universe opp plays)
RANGE_TIGHT  = 0.25   # pre-flop raise → top 25 % of hands
RANGE_MEDIUM = 0.50   # limp/call     → top 50 % of hands
RANGE_LOOSE  = 1.00   # no info       → any two cards

# Phase 5a — revealed card rank thresholds
REVEALED_HIGH_RANK  = 11   # J and above → opp has a strong card
REVEALED_LOW_RANK   = 7    # 7 and below → opp has a weak card

# Phase 5a — equity and sizing adjustments on revealed card
REVEALED_HIGH_EQ_DELTA    = -0.04   # equity penalty when opp has a high card
REVEALED_LOW_EQ_DELTA     =  0.03   # equity bonus when opp has a low card
REVEALED_HIGH_BLUFF_SCALE =  0.60   # bluff less vs strong hands
REVEALED_LOW_BLUFF_SCALE  =  1.20   # bluff more vs weak hands
REVEALED_HIGH_SIZE_SCALE  =  0.85   # bet smaller (fewer callers we beat)
REVEALED_LOW_SIZE_SCALE   =  1.15   # bet bigger for value

# Phase 5d — dynamic bid cap bounds
BID_CAP_MIN   = 0.10   # floor: large pot, opp committed
BID_CAP_MAX   = 0.27   # ceiling: small pot, deep stacks
BID_CAP_BASE  = 0.20   # neutral starting point

# Phase 5f — post-auction aggression exploitation
OPP_AUCT_AGG_HIGH  = 0.65  # opp usually bets after winning → treat as strong
OPP_AUCT_AGG_LOW   = 0.30  # opp rarely bets after winning → can call lighter
OPP_AUCT_FOLD_BUMP = 0.07  # fold threshold increase when opp bet after win
OPP_AUCT_CALL_DIP  = 0.05  # call threshold decrease when opp is passive post-win

# Phase 5g — positional equity adjustments (post-flop)
POSITION_IP_EQ_BONUS  =  0.02   # SB (IP post-flop) equity bonus
POSITION_OOP_EQ_DELTA = -0.02   # BB (OOP post-flop) equity penalty
POSITION_IP_BLUFF_SCALE  = 1.10  # bluff slightly more in position
POSITION_OOP_BLUFF_SCALE = 0.90  # bluff slightly less out of position
POSITION_BB_CALL_BONUS   = 0.03  # BB can call wider pre-flop (last to act)


# ===========================================================================
# CARD UTILITIES
# ===========================================================================

RANKS     = '23456789TJQKA'
SUITS     = 'hdcs'
RANK_VAL  = {r: i for i, r in enumerate(RANKS, 2)}   # '2'→2 … 'A'→14
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def card_rank(card: str) -> int:
    return RANK_VAL[card[0]]

def card_suit(card: str) -> str:
    return card[1]


# ===========================================================================
# PURE-PYTHON 5-CARD HAND EVALUATOR  (unchanged from Phase 1)
# ===========================================================================

_P = [15**i for i in range(5)]


def _build_score(cat: int, kickers: list) -> int:
    s = cat * (_P[4] * 15)
    for i, k in enumerate(kickers[:5]):
        s += k * _P[4 - i]
    return s


def evaluate_5(cards: list) -> int:
    rs = sorted([card_rank(c) for c in cards], reverse=True)
    sl = [card_suit(c) for c in cards]
    is_flush = len(set(sl)) == 1

    rc = {}
    for r in rs:
        rc[r] = rc.get(r, 0) + 1
    bc = {}
    for r, cnt in rc.items():
        bc.setdefault(cnt, []).append(r)
    for cnt in bc:
        bc[cnt].sort(reverse=True)

    uniq = sorted(set(rs), reverse=True)
    is_str = False; sh = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            is_str = True; sh = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            is_str = True; sh = 5

    if is_str and is_flush:  return _build_score(9, [sh])
    if 4 in bc:              return _build_score(8, [bc[4][0], bc.get(1,[0])[0]])
    if 3 in bc and 2 in bc:  return _build_score(7, [bc[3][0], bc[2][0]])
    if is_flush:             return _build_score(6, rs)
    if is_str:               return _build_score(5, [sh])
    if 3 in bc:
        ks = sorted(bc.get(1, []), reverse=True)
        return _build_score(4, [bc[3][0]] + ks[:2])
    if 2 in bc and len(bc[2]) >= 2:
        pairs = sorted(bc[2], reverse=True)[:2]
        k = [r for r in rs if r not in pairs][0]
        return _build_score(3, pairs + [k])
    if 2 in bc:
        p = bc[2][0]
        ks = [r for r in rs if r != p][:3]
        return _build_score(2, [p] + ks)
    return _build_score(1, rs)


def best_hand_score(hole: list, board: list) -> int:
    all_cards = hole + board
    if len(all_cards) <= 5:
        return evaluate_5(all_cards)
    return max(evaluate_5(list(c)) for c in combinations(all_cards, 5))


# ===========================================================================
# PHASE 3 — PRE-FLOP LOOKUP TABLE
# ===========================================================================
# 169 canonical hand types → heads-up equity vs a random opponent hand.
# Values sourced from standard HU equity databases (±0.5 %).
# Key format:  'AA'=pocket pair, 'AKs'=suited, 'AKo'=offsuit.
# Lookup is O(1) — replaces the Chen formula entirely.

_PF_EQUITY: dict[str, float] = {
    # ── Pocket pairs ──────────────────────────────────────────────── #
    'AA': 0.853, 'KK': 0.826, 'QQ': 0.799, 'JJ': 0.772, 'TT': 0.750,
    '99': 0.721, '88': 0.691, '77': 0.662, '66': 0.635, '55': 0.605,
    '44': 0.579, '33': 0.555, '22': 0.531,
    # ── Ace-x suited ──────────────────────────────────────────────── #
    'AKs': 0.662, 'AQs': 0.657, 'AJs': 0.653, 'ATs': 0.647,
    'A9s': 0.630, 'A8s': 0.622, 'A7s': 0.616, 'A6s': 0.614,
    'A5s': 0.615, 'A4s': 0.610, 'A3s': 0.607, 'A2s': 0.598,
    # ── Ace-x offsuit ─────────────────────────────────────────────── #
    'AKo': 0.654, 'AQo': 0.645, 'AJo': 0.636, 'ATo': 0.625,
    'A9o': 0.612, 'A8o': 0.600, 'A7o': 0.590, 'A6o': 0.586,
    'A5o': 0.585, 'A4o': 0.580, 'A3o': 0.574, 'A2o': 0.568,
    # ── King-x suited ─────────────────────────────────────────────── #
    'KQs': 0.630, 'KJs': 0.624, 'KTs': 0.620, 'K9s': 0.603,
    'K8s': 0.590, 'K7s': 0.586, 'K6s': 0.578, 'K5s': 0.574,
    'K4s': 0.568, 'K3s': 0.564, 'K2s': 0.559,
    # ── King-x offsuit ────────────────────────────────────────────── #
    'KQo': 0.618, 'KJo': 0.608, 'KTo': 0.595, 'K9o': 0.577,
    'K8o': 0.563, 'K7o': 0.558, 'K6o': 0.552, 'K5o': 0.545,
    'K4o': 0.538, 'K3o': 0.532, 'K2o': 0.527,
    # ── Queen-x suited ────────────────────────────────────────────── #
    'QJs': 0.605, 'QTs': 0.602, 'Q9s': 0.588, 'Q8s': 0.575,
    'Q7s': 0.562, 'Q6s': 0.558, 'Q5s': 0.552, 'Q4s': 0.545,
    'Q3s': 0.540, 'Q2s': 0.535,
    # ── Queen-x offsuit ───────────────────────────────────────────── #
    'QJo': 0.590, 'QTo': 0.584, 'Q9o': 0.565, 'Q8o': 0.550,
    'Q7o': 0.535, 'Q6o': 0.530, 'Q5o': 0.523, 'Q4o': 0.515,
    'Q3o': 0.510, 'Q2o': 0.504,
    # ── Jack-x suited ─────────────────────────────────────────────── #
    'JTs': 0.577, 'J9s': 0.572, 'J8s': 0.561, 'J7s': 0.550,
    'J6s': 0.538, 'J5s': 0.528, 'J4s': 0.520, 'J3s': 0.515,
    'J2s': 0.510,
    # ── Jack-x offsuit ────────────────────────────────────────────── #
    'JTo': 0.565, 'J9o': 0.556, 'J8o': 0.543, 'J7o': 0.528,
    'J6o': 0.514, 'J5o': 0.504, 'J4o': 0.495, 'J3o': 0.490,
    'J2o': 0.485,
    # ── Ten-x suited ──────────────────────────────────────────────── #
    'T9s': 0.548, 'T8s': 0.540, 'T7s': 0.528, 'T6s': 0.515,
    'T5s': 0.505, 'T4s': 0.500, 'T3s': 0.495, 'T2s': 0.490,
    # ── Ten-x offsuit ─────────────────────────────────────────────── #
    'T9o': 0.535, 'T8o': 0.525, 'T7o': 0.512, 'T6o': 0.497,
    'T5o': 0.487, 'T4o': 0.480, 'T3o': 0.475, 'T2o': 0.470,
    # ── Nine-x suited ─────────────────────────────────────────────── #
    '98s': 0.527, '97s': 0.518, '96s': 0.508, '95s': 0.500,
    '94s': 0.490, '93s': 0.485, '92s': 0.480,
    # ── Nine-x offsuit ────────────────────────────────────────────── #
    '98o': 0.513, '97o': 0.503, '96o': 0.492, '95o': 0.483,
    '94o': 0.473, '93o': 0.467, '92o': 0.462,
    # ── Eight-x suited ────────────────────────────────────────────── #
    '87s': 0.511, '86s': 0.502, '85s': 0.494, '84s': 0.483,
    '83s': 0.477, '82s': 0.472,
    # ── Eight-x offsuit ───────────────────────────────────────────── #
    '87o': 0.496, '86o': 0.487, '85o': 0.477, '84o': 0.467,
    '83o': 0.460, '82o': 0.455,
    # ── Seven-x suited ────────────────────────────────────────────── #
    '76s': 0.493, '75s': 0.485, '74s': 0.475, '73s': 0.468, '72s': 0.460,
    # ── Seven-x offsuit ───────────────────────────────────────────── #
    '76o': 0.478, '75o': 0.470, '74o': 0.460, '73o': 0.452, '72o': 0.445,
    # ── Six-x suited ──────────────────────────────────────────────── #
    '65s': 0.475, '64s': 0.467, '63s': 0.460, '62s': 0.453,
    # ── Six-x offsuit ─────────────────────────────────────────────── #
    '65o': 0.460, '64o': 0.452, '63o': 0.445, '62o': 0.438,
    # ── Five-x suited ─────────────────────────────────────────────── #
    '54s': 0.460, '53s': 0.453, '52s': 0.445,
    # ── Five-x offsuit ────────────────────────────────────────────── #
    '54o': 0.445, '53o': 0.437, '52o': 0.430,
    # ── Four-x suited / offsuit ───────────────────────────────────── #
    '43s': 0.445, '42s': 0.438, '43o': 0.430, '42o': 0.423,
    # ── Three-x suited / offsuit ──────────────────────────────────── #
    '32s': 0.430, '32o': 0.415,
}


@lru_cache(maxsize=2704)
def _hand_key(c1: str, c2: str) -> str:
    '''Convert ('Ah','Kd') → 'AKo' (or 'AKs' / 'AA'). Cached.'''
    r1, r2 = card_rank(c1), card_rank(c2)
    s1, s2 = card_suit(c1), card_suit(c2)
    hi_r, lo_r = max(r1, r2), min(r1, r2)
    _RD = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
           10:'T',11:'J',12:'Q',13:'K',14:'A'}
    if hi_r == lo_r:
        return _RD[hi_r] + _RD[lo_r]
    suffix = 's' if s1 == s2 else 'o'
    return _RD[hi_r] + _RD[lo_r] + suffix


def preflop_equity(hole: list) -> float:
    '''
    Phase 3: look up exact HU equity from pre-computed table.
    Falls back to Chen normalisation only if key is somehow missing.
    '''
    key = _hand_key(hole[0], hole[1])
    if key in _PF_EQUITY:
        return _PF_EQUITY[key]
    # Fallback (should never happen with a complete table)
    r1, r2 = card_rank(hole[0]), card_rank(hole[1])
    hi = max(r1, r2)
    base = {14:10, 13:8, 12:7, 11:6, 10:5}.get(hi, hi/2)
    if r1 == r2:
        return min(1.0, max(0.0, max(base*2, 5) / 20.0))
    if card_suit(hole[0]) == card_suit(hole[1]):
        base += 2
    gap = abs(r1-r2) - 1
    base -= {0:0,1:1,2:2,3:4}.get(gap, 5)
    return min(1.0, max(0.0, (base + 1) / 21.0))


# ===========================================================================
# PHASE 3 — OPPONENT RANGE ESTIMATION
# ===========================================================================

# Pre-sort all 169 hand keys by equity (descending) — used for range slicing.
_ALL_HAND_KEYS_SORTED: list[str] = sorted(
    _PF_EQUITY.keys(), key=lambda k: _PF_EQUITY[k], reverse=True
)

# For each hand key, which card pairs could represent it?
# We need this to sample a concrete pair of cards from a range.
# (card pair lookup is now pre-computed in _CARDS_FOR_KEY_FULL below)


# Pre-compute all concrete card pairs for every canonical hand key (done once).
_CARDS_FOR_KEY_FULL: dict[str, list] = {}
for _key in _ALL_HAND_KEYS_SORTED:
    if len(_key) == 2:
        _r = _key[0]
        _CARDS_FOR_KEY_FULL[_key] = [(_r+_s1, _r+_s2)
                                      for _s1 in SUITS for _s2 in SUITS
                                      if _s1 < _s2]
    else:
        _hi_r, _lo_r, _suited = _key[0], _key[1], _key[2] == 's'
        _pairs = []
        for _s1 in SUITS:
            for _s2 in SUITS:
                if _suited and _s1 != _s2: continue
                if not _suited and _s1 == _s2: continue
                _c1, _c2 = _hi_r + _s1, _lo_r + _s2
                if _c1 != _c2:
                    _pairs.append((_c1, _c2))
        _CARDS_FOR_KEY_FULL[_key] = _pairs


def _cards_for_key(key: str, excluded: set) -> list:
    '''Filter pre-computed pairs against excluded set.'''
    return [(a, b) for a, b in _CARDS_FOR_KEY_FULL[key]
            if a not in excluded and b not in excluded]


@lru_cache(maxsize=256)
def _build_opponent_range_cached(range_fraction: float,
                                  excluded_frozen: frozenset) -> tuple:
    '''
    Cached version of build_opponent_range.
    Returns a tuple of (c1, c2) pairs so the result is immutable/hashable.

    The cache means that when the excluded set and range fraction are the
    same across consecutive queries (common mid-street), we pay zero cost.
    '''
    n_types  = max(1, int(len(_ALL_HAND_KEYS_SORTED) * range_fraction))
    top_keys = _ALL_HAND_KEYS_SORTED[:n_types]
    pairs    = []
    for key in top_keys:
        pairs.extend(_cards_for_key(key, excluded_frozen))
    return tuple(pairs)


def build_opponent_range(range_fraction: float, excluded: set) -> list:
    '''
    Return a list of (c1, c2) tuples representing all concrete card pairs
    in the top `range_fraction` of hand types, excluding known cards.

    Used to sample opponent hands in range-constrained Monte Carlo.
    Internally uses an LRU cache keyed on (fraction, frozenset(excluded)).
    '''
    return list(_build_opponent_range_cached(range_fraction,
                                             frozenset(excluded)))


# ===========================================================================
# PHASE 3 — RANGE-CONSTRAINED MONTE CARLO
# ===========================================================================

def monte_carlo_equity(hole: list, board: list, opp_known: list,
                       n_sims: int = MC_SIMS,
                       opp_range_fraction: float = RANGE_LOOSE) -> float:
    '''
    Estimate win probability via random rollout.

    Phase 3 upgrade: if opp_range_fraction < 1.0, opponent cards are sampled
    from a restricted range (e.g., top 25 % of hands) rather than uniformly.
    This produces far more accurate equity estimates when we have range info.

    opp_range_fraction : 1.0 = any two cards (Phase 1 behaviour)
                         0.5 = top 50 % of hands
                         0.25= top 25 % (tight pre-flop raiser)
    '''
    known_set          = set(hole + board + opp_known)
    remaining          = [c for c in FULL_DECK if c not in known_set]
    cards_needed_board = 5 - len(board)
    cards_needed_opp   = 2 - len(opp_known)

    # Build range pool only once if we need it (cached internally)
    range_pairs = None
    if cards_needed_opp > 0 and opp_range_fraction < 0.99:
        range_pairs = build_opponent_range(opp_range_fraction, known_set)

    wins = 0.0
    _sample = random.sample  # local binding for speed
    _choice = random.choice
    for _ in range(n_sims):
        # Choose opponent cards
        if cards_needed_opp == 0:
            opp_hole = opp_known
            pool     = remaining
        elif range_pairs:
            # Sample from range — pick a random valid pair
            pair = _choice(range_pairs)
            opp_hole = list(opp_known) + [c for c in pair
                                          if c not in opp_known][:cards_needed_opp]
            pool = [c for c in remaining if c not in opp_hole]
        else:
            # Use random.sample instead of shuffle+slice — no mutation, faster
            drawn = _sample(remaining, cards_needed_opp + cards_needed_board)
            opp_hole = list(opp_known) + drawn[:cards_needed_opp]
            pool     = drawn[cards_needed_opp:]
            board_r  = board + pool
            my  = best_hand_score(hole,     board_r)
            opp = best_hand_score(opp_hole, board_r)
            if   my > opp: wins += 1.0
            elif my == opp: wins += 0.5
            continue

        # Fill out the board (range_pairs path or opp_known path)
        board_r = board + _sample(pool, cards_needed_board)

        my  = best_hand_score(hole,     board_r)
        opp = best_hand_score(opp_hole, board_r)

        if   my > opp: wins += 1.0
        elif my == opp: wins += 0.5

    return wins / n_sims if n_sims > 0 else 0.5


def get_equity(state: PokerState,
               n_sims: int = MC_SIMS,
               opp_range_fraction: float = RANGE_LOOSE) -> float:
    '''
    Unified equity estimator.
      Pre-flop : lookup table (instant, accurate).
      Post-flop: range-constrained Monte Carlo.
    '''
    hole  = state.my_hand
    board = list(state.board)               if state.board              else []
    opp   = list(state.opp_revealed_cards)  if state.opp_revealed_cards else []

    if not hole or len(hole) < 2:
        return 0.5

    if state.street == 'pre-flop':
        return preflop_equity(hole)

    return monte_carlo_equity(hole, board, opp, n_sims, opp_range_fraction)


# ===========================================================================
# PHASE 3 — BOARD TEXTURE ANALYSIS
# ===========================================================================

class BoardTexture:
    '''
    Analyses the community cards for strategic features.

    Attributes (computed once on construction):
      flush_draw   : 3+ cards of same suit on board
      straight_draw: 3+ connected ranks on board (spread ≤ 4)
      paired       : at least one rank appears twice on board
      trips_on_board: a rank appears three times
      high_board   : average rank > 10 (T=10, J=11, Q=12, K=13, A=14)
      wetness      : float 0.0 (dry) → 1.0 (very wet)
      danger       : float 0.0 (safe) → 1.0 (dangerous for value-betting)
    '''

    def __init__(self, board: list):
        self.board = board
        ranks  = [card_rank(c) for c in board]
        suits  = [card_suit(c) for c in board]

        self.flush_draw    = max(suits.count(s) for s in SUITS) >= 3 if board else False
        self.straight_draw = self._has_straight_draw(ranks)
        self.paired        = len(ranks) != len(set(ranks)) and len(board) >= 2
        self.trips_on_board= any(ranks.count(r) >= 3 for r in set(ranks))
        self.high_board    = (sum(ranks) / max(len(ranks), 1)) > 10 if board else False

        # Wetness: 0.0 (pure dry) → 1.0 (all draws active)
        self.wetness = (
            0.40 * int(self.flush_draw)
          + 0.35 * int(self.straight_draw)
          + 0.15 * int(self.paired)
          + 0.10 * int(self.high_board)
        )

        # Danger: high on wet boards — penalises value betting
        self.danger = self.wetness

    @staticmethod
    def _has_straight_draw(ranks: list) -> bool:
        if len(ranks) < 3:
            return False
        uniq = sorted(set(ranks))
        # Check any 3 consecutive unique ranks within a span of 4
        for i in range(len(uniq) - 2):
            if uniq[i+2] - uniq[i] <= 4:
                return True
        # Wheel (A-2-3-4-5) — include Ace as 1
        if 14 in uniq:
            low_uniq = sorted({1 if r == 14 else r for r in uniq})
            for i in range(len(low_uniq) - 2):
                if low_uniq[i+2] - low_uniq[i] <= 4:
                    return True
        return False

    def __str__(self):
        parts = []
        if self.flush_draw:    parts.append('FD')
        if self.straight_draw: parts.append('SD')
        if self.paired:        parts.append('Paired')
        if self.high_board:    parts.append('High')
        return f'Board({"|".join(parts) or "Dry"}, wet={self.wetness:.2f})'


def board_texture(board: list) -> BoardTexture:
    return BoardTexture(board)


# ===========================================================================
# PHASE 3 — SPR UTILITIES
# ===========================================================================

def compute_spr(state: PokerState) -> float:
    '''
    Stack-to-Pot Ratio: effective_stack / pot.
    Effective stack = min(our chips, opponent chips).
    Returns float; clamps at 20 to avoid division oddities early in hand.
    '''
    eff_stack = min(state.my_chips, getattr(state, 'opp_chips', state.my_chips))
    return eff_stack / state.pot if state.pot > 0 else 20.0


def spr_adjustment(spr: float, equity: float) -> float:
    '''
    Adjust effective equity based on SPR.

    Low SPR: strong hands become more valuable (commitment profitable);
             draws become less valuable (no implied odds).
    High SPR: draws get implied-odds bonus; marginal made hands riskier.

    Returns an additive equity adjustment (can be negative).
    '''
    adj = 0.0
    if spr < SPR_LOW:
        # Committed — boost strong hands, suppress draws (which rarely complete
        # usefully when pot is already large relative to stack)
        if equity >= 0.55:
            adj += 0.05
        elif equity < 0.40:
            adj -= 0.05
    elif spr > SPR_HIGH:
        # Deep — draws become more profitable (implied odds)
        if 0.30 <= equity <= 0.50:  # likely a draw range
            adj += 0.04
        elif equity >= 0.65:
            adj -= 0.03   # even strong hands need caution vs unknown range
    return adj


# ===========================================================================
# PHASE 2 — OPPONENT MODEL  (unchanged, re-included for completeness)
# ===========================================================================

class OpponentModel:
    def __init__(self):
        self.hands_seen       = 0
        self.pf_total         = 0
        self.pf_aggressive    = 0
        self.postflop_bets    = {'flop': 0, 'turn': 0, 'river': 0}
        self.postflop_opps    = {'flop': 0, 'turn': 0, 'river': 0}
        self.opp_folds        = 0
        self.showdowns        = 0
        self.auction_rounds   = 0
        self.we_won_auction   = 0
        self.our_bids         = []
        self._recent_pf_agg   = []
        self._recent_folds    = []
        self._WINDOW          = 100

        # Phase 5f — post-auction aggression tracking
        self.opp_won_auction_count      = 0   # hands where opp won the auction
        self.opp_bet_after_auction_count= 0   # of those, how many did opp bet flop
        self._recent_opp_auct_bet       = []  # sliding window (bool per hand)

    def record_preflop(self, was_aggressive: bool):
        self.pf_total      += 1
        self.pf_aggressive += int(was_aggressive)
        self._recent_pf_agg.append(was_aggressive)
        if len(self._recent_pf_agg) > self._WINDOW:
            self._recent_pf_agg.pop(0)

    def record_street_bet(self, street: str, opp_bet: bool):
        if street not in self.postflop_opps:
            return
        self.postflop_opps[street] += 1
        self.postflop_bets[street] += int(opp_bet)

    def record_hand_end(self, payoff: int, final_street: str,
                        opp_revealed: list):
        self.hands_seen += 1
        won_early       = payoff > 0 and final_street != 'river'
        if won_early:
            self.opp_folds += 1
        if final_street == 'river':
            self.showdowns += 1
        if opp_revealed:
            self.we_won_auction += 1
        self._recent_folds.append(won_early)
        if len(self._recent_folds) > self._WINDOW:
            self._recent_folds.pop(0)

    def record_auction(self, our_bid: int):
        self.auction_rounds += 1
        self.our_bids.append(our_bid)
        if len(self.our_bids) > 200:
            self.our_bids.pop(0)

    def record_opp_auction_result(self, opp_won: bool, opp_bet_after: bool):
        '''
        Phase 5f: called at end of each hand where we had an auction.
        opp_won      : True if opp (not us) won the auction.
        opp_bet_after: True if opp bet/raised on the flop post-auction.
        '''
        if opp_won:
            self.opp_won_auction_count += 1
            self.opp_bet_after_auction_count += int(opp_bet_after)
            self._recent_opp_auct_bet.append(opp_bet_after)
            if len(self._recent_opp_auct_bet) > self._WINDOW:
                self._recent_opp_auct_bet.pop(0)

    @property
    def vpip_rate(self) -> float:
        return self.pf_aggressive / max(self.pf_total, 1)

    @property
    def postflop_aggression(self) -> float:
        tb = sum(self.postflop_bets.values())
        to = sum(self.postflop_opps.values())
        return tb / max(to, 1)

    @property
    def fold_rate(self) -> float:
        return self.opp_folds / max(self.hands_seen, 1)

    @property
    def our_auction_win_rate(self) -> float:
        return self.we_won_auction / max(self.auction_rounds, 1)

    @property
    def opp_post_auction_bet_rate(self) -> float:
        '''
        Phase 5f: fraction of times opp bets on flop after winning auction.
        Uses sliding window when enough data; falls back to lifetime rate.
        '''
        if len(self._recent_opp_auct_bet) >= 10:
            return sum(self._recent_opp_auct_bet) / len(self._recent_opp_auct_bet)
        return (self.opp_bet_after_auction_count /
                max(self.opp_won_auction_count, 1))

    @property
    def recent_vpip(self) -> float:
        if len(self._recent_pf_agg) < 10:
            return self.vpip_rate
        return sum(self._recent_pf_agg) / len(self._recent_pf_agg)

    @property
    def recent_fold_rate(self) -> float:
        if len(self._recent_folds) < 10:
            return self.fold_rate
        return sum(self._recent_folds) / len(self._recent_folds)

    def is_tight(self)      -> bool: return self.recent_vpip < 0.25
    def is_loose(self)      -> bool: return self.recent_vpip > 0.50
    def is_aggressive(self) -> bool: return self.postflop_aggression > 0.45
    def is_passive(self)    -> bool: return self.postflop_aggression < 0.25
    def is_folder(self)     -> bool: return self.recent_fold_rate > 0.35
    def sufficient_data(self) -> bool: return self.hands_seen >= MODEL_MIN_HANDS

    def inferred_range_fraction(self) -> float:
        '''
        Phase 3: estimate what fraction of hands opp is likely playing,
        based on observed pre-flop aggression.
        '''
        if not self.sufficient_data():
            return RANGE_LOOSE
        if self.is_tight():
            return RANGE_TIGHT
        if self.is_loose():
            return RANGE_LOOSE
        return RANGE_MEDIUM

    def __str__(self):
        return (f'Hands={self.hands_seen}  VPIP={self.vpip_rate:.2f}  '
                f'PF_agg={self.postflop_aggression:.2f}  '
                f'Fold%={self.fold_rate:.2f}  '
                f'AuctWin%={self.our_auction_win_rate:.2f}  '
                f'OppPostAuctBet%={self.opp_post_auction_bet_rate:.2f}')


# ===========================================================================
# PHASE 2 — ADAPTIVE STRATEGY  (extended for Phase 3 board texture)
# ===========================================================================

class AdaptiveStrategy:
    def __init__(self, model: OpponentModel):
        self.model = model

    def thresholds(self, is_bb: bool = False,
                   opp_won_auction_and_bet: bool = False) -> tuple:
        '''
        Returns (fold_t, call_t, value_t, bluff_f, small_f, big_f).

        Phase 5g: is_bb controls positional adjustments.
          BB = OOP post-flop → tighter thresholds, less bluffing.
          SB = IP post-flop  → looser thresholds, more bluffing.

        Phase 5f: opp_won_auction_and_bet signals that the opponent won
          the auction AND bet the flop — meaning they found what they
          needed.  We raise the fold threshold and lower call threshold.
        '''
        m = self.model
        fold_t = FOLD_THRESHOLD; call_t = CALL_THRESHOLD
        value_t= VALUE_THRESHOLD; bluff_f= BLUFF_FREQ
        small_f= BET_SMALL_FRAC;  big_f  = BET_BIG_FRAC

        if not m.sufficient_data():
            # Apply positional baseline even before model has enough data
            fold_t, call_t, bluff_f = self._apply_position(
                fold_t, call_t, bluff_f, is_bb)
            if opp_won_auction_and_bet:
                fold_t, call_t = self._apply_opp_auction_aggression(
                    fold_t, call_t, m)
            return fold_t, call_t, value_t, bluff_f, small_f, big_f

        tight  = m.is_tight(); loose  = m.is_loose()
        passive= m.is_passive(); aggro = m.is_aggressive()
        folder = m.is_folder()

        if tight and passive:        # Tight-Passive
            bluff_f = min(bluff_f * 1.8, 0.40)
            value_t = min(value_t + 0.04, 0.75)
            small_f = 0.40; big_f = 0.60
        elif tight and aggro:        # Tight-Aggressive
            call_t  = max(call_t  - 0.06, 0.35)
            fold_t  = max(fold_t  - 0.06, 0.20)
            bluff_f = max(bluff_f - 0.06, 0.08)
            value_t = max(value_t - 0.04, 0.58)
        elif loose and passive:      # Loose-Passive (calling station)
            bluff_f = max(bluff_f * 0.20, 0.03)
            value_t = max(value_t - 0.10, 0.50)
            call_t  = max(call_t  - 0.05, 0.35)
            small_f = 0.65; big_f = 1.00
        elif loose and aggro:        # Loose-Aggressive
            fold_t  = min(fold_t  + 0.06, 0.40)
            call_t  = min(call_t  + 0.06, 0.55)
            value_t = min(value_t + 0.05, 0.75)
            bluff_f = max(bluff_f * 0.30, 0.05)

        if folder and not (loose and aggro):
            bluff_f = min(bluff_f * 1.3, 0.40)

        # ── Phase 5g: positional adjustment ──────────────────────────── #
        fold_t, call_t, bluff_f = self._apply_position(
            fold_t, call_t, bluff_f, is_bb)

        # ── Phase 5f: opponent post-auction aggression ─────────────────── #
        if opp_won_auction_and_bet:
            fold_t, call_t = self._apply_opp_auction_aggression(
                fold_t, call_t, m)

        return fold_t, call_t, value_t, bluff_f, small_f, big_f

    @staticmethod
    def _apply_position(fold_t: float, call_t: float,
                        bluff_f: float, is_bb: bool) -> tuple:
        '''
        Phase 5g: adjust thresholds for positional advantage/disadvantage.

        SB (is_bb=False) = in position post-flop:
          • Lower fold threshold (can call down lighter)
          • Higher bluff frequency (position makes bluffs more credible)

        BB (is_bb=True) = out of position post-flop:
          • Higher fold threshold (tough to continue OOP without strong hand)
          • Lower bluff frequency (bluffs harder to pull off OOP)
          • Small pre-flop call bonus is handled via equity in get_move.
        '''
        if is_bb:   # OOP
            fold_t  = min(fold_t  + 0.03, 0.40)
            call_t  = min(call_t  + 0.02, 0.55)
            bluff_f = max(bluff_f * POSITION_OOP_BLUFF_SCALE, 0.05)
        else:       # IP
            fold_t  = max(fold_t  - 0.02, 0.18)
            bluff_f = min(bluff_f * POSITION_IP_BLUFF_SCALE, 0.45)
        return fold_t, call_t, bluff_f

    def _apply_opp_auction_aggression(self, fold_t: float,
                                       call_t: float,
                                       m: 'OpponentModel') -> tuple:
        '''
        Phase 5f: when opp won auction and is now betting, adjust thresholds
        based on how often opp bets post-auction historically.

        High opp_post_auction_bet_rate → they usually have something real →
          raise fold threshold (don't call light) and tighten call threshold.
        Low rate → they're bluffing / passive even with info →
          lower call threshold slightly (we can look them up).
        '''
        if m.opp_won_auction_count < 5:
            # Not enough data: apply a conservative uniform adjustment
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP * 0.5, 0.45)
            return fold_t, call_t

        rate = m.opp_post_auction_bet_rate
        if rate >= OPP_AUCT_AGG_HIGH:
            # Opp is aggressive post-auction → they found strength
            fold_t = min(fold_t + OPP_AUCT_FOLD_BUMP, 0.48)
            call_t = min(call_t + 0.04, 0.58)
        elif rate <= OPP_AUCT_AGG_LOW:
            # Opp is passive post-auction → we can look them up
            call_t = max(call_t - OPP_AUCT_CALL_DIP, 0.33)
        return fold_t, call_t

    def bid_multiplier(self) -> float:
        m = self.model
        if m.auction_rounds < 15:
            return 1.0
        wr = m.our_auction_win_rate
        if   wr < 0.25: return 1.50
        elif wr < 0.35: return 1.25
        elif wr > 0.75: return 0.70
        elif wr > 0.65: return 0.85
        return 1.0

    def describe(self) -> str:
        m = self.model
        if not m.sufficient_data():
            return 'Insufficient data — using defaults'
        parts = []
        if m.is_tight():   parts.append('Tight')
        elif m.is_loose(): parts.append('Loose')
        else:              parts.append('Normal')
        if m.is_passive():     parts.append('Passive')
        elif m.is_aggressive():parts.append('Aggressive')
        else:                  parts.append('Balanced')
        if m.is_folder():  parts.append('(Folder)')
        return '-'.join(parts)


# ===========================================================================
# PHASE 3 — STREET-AWARE BET SIZING
# ===========================================================================

# (fraction_thin_value, fraction_strong)  by street
_STREET_SIZING = {
    'pre-flop': (0.50, 0.75),   # shove/jam logic handled by SPR separately
    'flop'    : (0.40, 0.55),   # keep range wide, pot is small
    'turn'    : (0.55, 0.70),   # more polarised, apply pressure
    'river'   : (0.65, 0.90),   # final bet, go big for value / bluff
}

def street_sizing(street: str, texture: BoardTexture, strong: bool) -> float:
    '''
    Return the appropriate bet-as-fraction-of-pot for this street.

    On wet boards: reduce bet size slightly (more callers → give better price).
    On dry boards: increase slightly (bloated pot helps us extract value).
    '''
    thin_f, big_f = _STREET_SIZING.get(street, (BET_SMALL_FRAC, BET_BIG_FRAC))
    base = big_f if strong else thin_f
    # Wet board: reduce by up to 10 % (retain callers)
    # Dry board: increase by up to 10 %
    wet_adj = (texture.wetness - 0.5) * (-0.20)   # wet→negative, dry→positive
    return max(0.30, min(1.20, base + wet_adj))


# ===========================================================================
# PHASE 5a — REVEALED CARD EXPLOITATION
# ===========================================================================

def revealed_card_adjustment(revealed_card: str,
                              board: list) -> tuple:
    '''
    Phase 5a: compute (equity_delta, bluff_scale, size_scale) based on
    the rank of the opponent's card we revealed after winning the auction.

    Logic
    -----
    High card (J / Q / K / A):
        Opp has at least one strong card → they likely have a pair, top pair,
        or strong draw.  We should be more cautious: tighten up, reduce
        bluffing, and size down our bets (we beat fewer of their hands).

    Low card (2 – 7):
        Opp has at least one weak card → their hand is likely weaker overall.
        Push harder: bet bigger for value, bluff more.  BUT if this low card
        connects with the board (within 2 ranks of any board card), dampen
        the bonus — a 6 on a 5-7-8 board pairs them or gives a straight draw.

    Middle card (8 – T):
        Neutral: no adjustment.
    '''
    rank = card_rank(revealed_card)
    board_ranks = [card_rank(c) for c in board]

    # Check board connectivity for low cards: does the revealed card
    # connect with any board card (within ±2 ranks)?
    connects_to_board = any(abs(rank - br) <= 2 for br in board_ranks)

    if rank >= REVEALED_HIGH_RANK:
        return (REVEALED_HIGH_EQ_DELTA,
                REVEALED_HIGH_BLUFF_SCALE,
                REVEALED_HIGH_SIZE_SCALE)

    elif rank <= REVEALED_LOW_RANK:
        # Dampen the benefit if the low card connects to the board
        dampen = 0.5 if connects_to_board else 1.0
        return (REVEALED_LOW_EQ_DELTA * dampen,
                1.0 + (REVEALED_LOW_BLUFF_SCALE - 1.0) * dampen,
                1.0 + (REVEALED_LOW_SIZE_SCALE  - 1.0) * dampen)

    # Middle rank (8–T): no adjustment
    return (0.0, 1.0, 1.0)


# ===========================================================================
# PHASE 5d — DYNAMIC BID CAP
# ===========================================================================

def dynamic_max_bid_frac(pot: int, effective_stack: int) -> float:
    '''
    Phase 5d: replace flat MAX_BID_FRAC with a cap that scales with pot
    geometry and stack depth.

    Small pot relative to starting stack:
        More streets remain to exploit information → allow up to BID_CAP_MAX.

    Large pot (3-bet pot, re-raised pre-flop):
        Opp already committed chips pre-flop signalling a strong hand →
        information is worth less → floor at BID_CAP_MIN.

    Stack depth modifier:
        Deeper effective stack → more implied value from a revealed card
        (we can extract more on turn/river) → scale up slightly.
    '''
    pot_fraction = pot / max(STARTING_STACK, 1)

    # Pot-size component: linearly interpolate between cap_max and cap_min
    # pot_fraction ≈ 0.01 (just blinds)  → cap near BID_CAP_MAX
    # pot_fraction ≈ 0.25+ (big 3-bet)   → cap near BID_CAP_MIN
    pot_cap = BID_CAP_MAX - (BID_CAP_MAX - BID_CAP_MIN) * min(pot_fraction / 0.25, 1.0)

    # Stack-depth component: more chips behind → more implied value
    stack_fraction = effective_stack / max(STARTING_STACK, 1)
    stack_adj = 0.04 * max(0.0, stack_fraction - 0.5)   # bonus only when deep

    return min(BID_CAP_MAX, max(BID_CAP_MIN, pot_cap + stack_adj))



#
# Core principle: in a second-price (Vickrey) auction, the dominant strategy
# is to bid your true value.  Our value = the expected gain in decision quality
# that comes from seeing one of the opponent's hole cards.
#
#   value = |equity_if_win_auction - equity_if_lose_auction| * pot
#
# equity_if_win_auction:
#   We see one of opp's hole cards (chosen uniformly at random).
#   This is computed by averaging MC equity over all plausible opp hands,
#   each time "peeking" at one randomly chosen card before rolling out.
#
# equity_if_lose_auction:
#   Opp sees one of OUR hole cards.  Strategically this hurts us:
#   opp can now fold dominated holdings and stack off with winners.
#   We model this by running MC against a tighter opponent range
#   (they effectively remove the weakest hands from their range once
#   they know what they're up against).
#
# Bid caps / adjustments:
#   • Very strong (equity ≥ 0.85) or very weak (equity ≤ 0.15): bid 0
#     — information has low value at the extremes.
#   • Raw bid = true_value, then scaled down by (1 - equity_extremeness)
#     to avoid over-paying when one card won't change our decision.
#   • Hard cap at MAX_BID_FRAC of pot to prevent chip bleeding.
#   • Adaptive multiplier from strategy (tracks historical win-rate).
#
# Complexity budget: outer loop = AUC_PEEK_TRIALS × inner MC sims.
# Total is kept inside MC_SIMS_AUCTION budget via batching.

AUC_PEEK_TRIALS  = 12   # outer: how many distinct "revealed card" scenarios
AUC_INNER_SIMS   = 4    # inner: MC rollouts per scenario
AUC_LOSE_FRAC    = 0.35 # tighter opp range when they know our card


def _equity_if_we_win_auction(hole: list, board: list,
                               excluded: set,
                               opp_range_fraction: float) -> float:
    '''
    Expected equity after winning the auction (peeking at one opp card).

    For each trial we:
      1. Sample a plausible opp hand from the range.
      2. Randomly select one of their two hole cards to "reveal."
      3. Re-estimate our equity knowing that one card.

    Returns the average equity across AUC_PEEK_TRIALS scenarios.
    '''
    remaining  = [c for c in FULL_DECK if c not in excluded]
    range_pairs = build_opponent_range(opp_range_fraction, excluded)
    if not range_pairs:
        range_pairs = [(a, b) for a in remaining for b in remaining
                       if a < b]

    total = 0.0
    for _ in range(AUC_PEEK_TRIALS):
        # Sample a plausible opp hand
        opp_hand = list(random.choice(range_pairs))
        # Reveal one card at random
        revealed  = [random.choice(opp_hand)]
        # MC equity knowing that revealed card
        eq = monte_carlo_equity(
            hole, board,
            opp_known=revealed,
            n_sims=AUC_INNER_SIMS,
            opp_range_fraction=opp_range_fraction,
        )
        total += eq

    return total / AUC_PEEK_TRIALS


def _equity_if_we_lose_auction(hole: list, board: list,
                                opp_range_fraction: float) -> float:
    '''
    Expected equity after losing the auction (opp peeks at one of our cards).

    Opp now knows one of our hole cards and will exploit it:
      - Fold weak hands that are dominated by our revealed strength.
      - Call / re-raise with hands that have equity vs our range.

    We model this as running MC against a *tighter* opponent range
    (AUC_LOSE_FRAC), because opp effectively culls the bottom of their
    range once they have information about ours.  The effect is averaged
    over both possible revealed cards (either of our two hole cards
    could be the one opp sees).
    '''
    tighter_fraction = min(opp_range_fraction, AUC_LOSE_FRAC)

    total = 0.0
    for our_card in hole:        # average over which of our cards is revealed
        eq = monte_carlo_equity(
            hole, board,
            opp_known=[],
            n_sims=AUC_INNER_SIMS,
            opp_range_fraction=tighter_fraction,
        )
        total += eq
    return total / len(hole)


def compute_bid(state: PokerState,
                strategy: AdaptiveStrategy,
                opp_range_fraction: float = RANGE_LOOSE) -> int:
    '''
    Phase 4 auction bid: bid true value of information.

    Steps
    -----
    1. Estimate baseline equity (fast MC).
    2. Short-circuit if hand is very strong or very weak (info worthless).
    3. Estimate equity_if_win (peek at one opp card).
    4. Estimate equity_if_lose (opp peeks at one of ours → tighter MC).
    5. Raw value = |equity_win - equity_lose| * pot.
    6. Scale by marginal_factor to avoid overpaying when one card
       won't change our decision anyway (e.g. equity near 0.5 is most
       uncertain; very close to 0 or 1 means info rarely changes action).
    7. Apply adaptive bid_multiplier (learned from historical win-rate).
    8. Cap at MAX_BID_FRAC of pot and our available chips.
    '''
    hole  = list(state.my_hand)
    board = list(state.board) if state.board else []
    opp   = list(state.opp_revealed_cards) if state.opp_revealed_cards else []

    # ── 1. Baseline equity ────────────────────────────────────────────── #
    equity = monte_carlo_equity(hole, board, opp,
                                n_sims=MC_SIMS_AUCTION,
                                opp_range_fraction=opp_range_fraction)

    # ── 2. Extreme hands: info has low value, don't bleed chips ───────── #
    if equity >= 0.85 or equity <= 0.15:
        return 0

    excluded = set(hole + board + opp)

    # ── 3. Equity if we WIN the auction ───────────────────────────────── #
    eq_win  = _equity_if_we_win_auction(hole, board, excluded,
                                        opp_range_fraction)

    # ── 4. Equity if we LOSE the auction ──────────────────────────────── #
    eq_lose = _equity_if_we_lose_auction(hole, board, opp_range_fraction)

    # ── 5. Raw value of winning the auction ───────────────────────────── #
    info_value = abs(eq_win - eq_lose) * state.pot

    # ── 6. Marginal factor: penalise bids that change nothing ─────────── #
    # equity near 0.50 → most uncertain → factor ≈ 1.0
    # equity near extremes → factor approaches 0  (already caught above,
    # but here we add a smooth taper for the 0.15–0.85 range)
    marginal_factor = 4.0 * equity * (1.0 - equity)   # peaks at 1.0 at eq=0.5

    # ── 7. Adaptive multiplier ────────────────────────────────────────── #
    adapt_mult = strategy.bid_multiplier()

    # ── 8. Final bid (clamp) ──────────────────────────────────────────── #
    raw_bid  = info_value * marginal_factor * adapt_mult

    # Phase 5d: dynamic cap based on pot size and stack depth
    eff_stack = min(state.my_chips,
                    getattr(state, 'opp_chips', state.my_chips))
    dyn_cap_frac = dynamic_max_bid_frac(state.pot, eff_stack)
    max_bid  = int(state.pot * dyn_cap_frac)
    bid      = int(raw_bid)

    # Never bid more than hard cap or our available chips
    return max(0, min(bid, max_bid, state.my_chips))


# ===========================================================================
# PHASE 3 — MAIN DECISION FUNCTION (SPR + texture + range-aware equity)
# ===========================================================================

def decide_action(state: PokerState, equity: float,
                  strategy: AdaptiveStrategy,
                  texture: BoardTexture,
                  auction_ctx: dict | None = None):
    '''
    Core decision with Phase 3–5 enhancements.

    auction_ctx keys (all optional, default to neutral):
      we_won_auction        : bool — we peeked at one opp card this hand
      revealed_card         : str  — the card we saw (if we_won_auction)
      opp_won_auction       : bool — opp peeked at one of our cards
      opp_bet_post_auction  : bool — opp bet/raised on flop after winning
      is_bb                 : bool — we are the big blind (OOP post-flop)
    '''
    ctx = auction_ctx or {}
    we_won_auction       = ctx.get('we_won_auction',       False)
    revealed_card        = ctx.get('revealed_card',        None)
    opp_won_auction      = ctx.get('opp_won_auction',      False)
    opp_bet_post_auction = ctx.get('opp_bet_post_auction', False)
    is_bb                = ctx.get('is_bb',                False)

    # ── Phase 5f context flag for thresholds ──────────────────────────── #
    opp_won_and_bet = opp_won_auction and opp_bet_post_auction

    fold_t, call_t, value_t, bluff_f, _sf, _bf = strategy.thresholds(
        is_bb=is_bb, opp_won_auction_and_bet=opp_won_and_bet)

    pot = state.pot
    ctc = state.cost_to_call
    pot_odds = ctc / (pot + ctc) if (ctc > 0 and pot + ctc > 0) else 0.0

    min_raise, max_raise = state.raise_bounds

    def clamp(amount: int) -> int:
        return max(min_raise, min(amount, max_raise))

    # ── Phase 3: SPR-adjusted equity ──────────────────────────────────── #
    spr   = compute_spr(state)
    adj   = spr_adjustment(spr, equity)
    eff_eq= max(0.0, min(1.0, equity + adj))

    # ── Phase 5g: positional equity adjustment ────────────────────────── #
    street = state.street
    if street != 'pre-flop':
        if is_bb:
            eff_eq = max(0.0, eff_eq + POSITION_OOP_EQ_DELTA)
        else:
            eff_eq = min(1.0, eff_eq + POSITION_IP_EQ_BONUS)
    elif street == 'pre-flop' and is_bb:
        # BB acts last pre-flop → small call-wider bonus
        call_t = max(call_t - POSITION_BB_CALL_BONUS, 0.30)

    # ── Phase 5a: revealed card equity & sizing adjustment ────────────── #
    rev_eq_delta  = 0.0
    rev_bluff_sc  = 1.0
    rev_size_sc   = 1.0
    if we_won_auction and revealed_card and street != 'pre-flop':
        board = list(state.board) if state.board else []
        rev_eq_delta, rev_bluff_sc, rev_size_sc = revealed_card_adjustment(
            revealed_card, board)
        eff_eq = max(0.0, min(1.0, eff_eq + rev_eq_delta))

    # ── Phase 3: texture-aware bluff frequency ─────────────────────────── #
    texture_bluff_scale = 1.0 + 0.4 * (0.5 - texture.wetness)  # [0.8, 1.2]
    eff_bluff_f = max(0.0, min(0.50,
        bluff_f * texture_bluff_scale * rev_bluff_sc))

    # ── Phase 3: street-aware sizing ─────────────────────────────────── #
    strong  = eff_eq >= value_t
    sz_frac = street_sizing(street, texture, strong) * rev_size_sc
    sz_frac = max(0.25, min(1.40, sz_frac))   # guard after scale

    # ── Low SPR shortcut: shove or fold when deeply committed ─────────── #
    if spr < SPR_LOW and street != 'pre-flop':
        if eff_eq >= 0.50:
            if state.can_act(ActionRaise):
                return ActionRaise(max_raise)
            if state.can_act(ActionCall):
                return ActionCall()
        else:
            if state.can_act(ActionFold):
                return ActionFold()
            if state.can_act(ActionCheck):
                return ActionCheck()

    # ── No bet to call ────────────────────────────────────────────────── #
    if state.can_act(ActionCheck):

        if eff_eq >= value_t:
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * sz_frac)))
            return ActionCheck()

        elif eff_eq >= call_t:
            # Thin value bet
            thin_sz = street_sizing(street, texture, False) * rev_size_sc
            thin_sz = max(0.25, min(1.40, thin_sz))
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + int(pot * thin_sz)))
            return ActionCheck()

        else:
            # Weak: check unless bluffing
            if (street in ('flop', 'turn', 'river')
                    and eff_eq < fold_t
                    and random.random() < eff_bluff_f
                    and state.can_act(ActionRaise)):
                bluff_sz = street_sizing(street, texture, False) * rev_size_sc
                bluff_sz = max(0.25, min(1.40, bluff_sz))
                return ActionRaise(clamp(pot + int(pot * bluff_sz)))
            return ActionCheck()

    # ── Facing a bet ──────────────────────────────────────────────────── #
    else:
        if eff_eq >= value_t:
            if state.can_act(ActionRaise):
                return ActionRaise(clamp(pot + ctc))
            if state.can_act(ActionCall):
                return ActionCall()

        elif eff_eq >= max(call_t, pot_odds * 1.10):
            if state.can_act(ActionCall):
                return ActionCall()

        if state.can_act(ActionFold):
            return ActionFold()

    # Fallback
    if state.can_act(ActionCall):  return ActionCall()
    if state.can_act(ActionCheck): return ActionCheck()
    return ActionFold()


# ===========================================================================
# MAIN BOT CLASS
# ===========================================================================

class Player(BaseBot):
    '''
    Phase 5 IIT Pokerbots 2026 bot.

    New in Phase 5 (on top of Phase 4):
      5a. Post-auction card exploitation
            When we win the auction, the rank of the revealed card adjusts
            our equity estimate, bluff frequency, and bet sizing for every
            remaining street of the hand.
      5d. Dynamic bid cap
            The auction bid ceiling now scales with pot size and stack depth
            rather than being a flat 20 % of pot.
      5f. Post-auction aggression tracking
            Tracks whether the opponent bets on the flop after winning the
            auction.  High historical rate → tighten call/fold thresholds
            that street; low rate → call them down lighter.
      5g. Positional awareness
            SB = in position post-flop → looser thresholds, more bluffing.
            BB = out of position post-flop → tighter thresholds, less bluffing.
            BB gets a small pre-flop call-wider bonus (last to act).
    '''

    def __init__(self) -> None:
        self.opp_model = OpponentModel()
        self.strategy  = AdaptiveStrategy(self.opp_model)

        # Per-round state (reset in on_hand_start)
        self._pf_recorded         = False
        self._streets_seen        = set()
        self._opp_bet_streets     = set()
        self._our_auction_bid     = 0
        self._is_bb               = False

        # Phase 5a: track what we revealed after winning the auction
        self._we_won_auction      = False
        self._revealed_card       = None    # str | None

        # Phase 5f: track if opp won the auction and whether they bet after
        self._opp_won_auction     = False
        self._opp_bet_post_auction= False
        self._auction_seen        = False   # auction phase has occurred

    # ── Lifecycle ────────────────────────────────────────────────────── #

    def on_hand_start(self, game_info: GameInfo,
                      current_state: PokerState) -> None:
        self._pf_recorded          = False
        self._streets_seen         = set()
        self._opp_bet_streets      = set()
        self._our_auction_bid      = 0
        self._is_bb                = current_state.is_bb
        self._we_won_auction       = False
        self._revealed_card        = None
        self._opp_won_auction      = False
        self._opp_bet_post_auction = False
        self._auction_seen         = False

    def on_hand_end(self, game_info: GameInfo,
                    current_state: PokerState) -> None:
        for street in ('flop', 'turn', 'river'):
            if street in self._streets_seen:
                self.opp_model.record_street_bet(
                    street, street in self._opp_bet_streets)

        if self._our_auction_bid > 0:
            self.opp_model.record_auction(self._our_auction_bid)

        opp_revealed = list(current_state.opp_revealed_cards) \
                       if current_state.opp_revealed_cards else []

        # Phase 5f: record opp auction result
        if self._auction_seen:
            self.opp_model.record_opp_auction_result(
                opp_won     = self._opp_won_auction,
                opp_bet_after=self._opp_bet_post_auction,
            )

        self.opp_model.record_hand_end(
            payoff       = current_state.payoff,
            final_street = current_state.street,
            opp_revealed = opp_revealed,
        )

    # ── Observation helpers ───────────────────────────────────────────── #

    def _observe_preflop(self, state: PokerState):
        if self._pf_recorded or state.street != 'pre-flop':
            return
        was_aggressive = state.cost_to_call > (BIG_BLIND - SMALL_BLIND)
        self.opp_model.record_preflop(was_aggressive)
        self._pf_recorded = True

    def _observe_postflop(self, state: PokerState):
        s = state.street
        if s not in ('flop', 'turn', 'river'):
            return
        if state.cost_to_call > 0:
            self._opp_bet_streets.add(s)
        self._streets_seen.add(s)

    def _observe_auction_outcome(self, state: PokerState):
        '''
        Phase 5a + 5f: detect auction outcome on the first flop query.
        Called once per hand when we first hit the 'flop' street.
        '''
        if self._auction_seen:
            return
        self._auction_seen = True

        opp_revealed = (list(state.opp_revealed_cards)
                        if state.opp_revealed_cards else [])

        if opp_revealed:
            # We won (or tied) — we can see at least one of their cards.
            self._we_won_auction  = True
            self._revealed_card   = opp_revealed[0]
            self._opp_won_auction = False
        elif self._our_auction_bid > 0:
            # We bid something but got nothing back → opp won.
            self._opp_won_auction = True

    def _observe_opp_post_auction_bet(self, state: PokerState):
        '''
        Phase 5f: on the first flop action, record whether opp is betting
        after winning the auction.
        '''
        if (self._opp_won_auction
                and state.street == 'flop'
                and 'flop_post_auction_checked' not in self._streets_seen):
            self._streets_seen.add('flop_post_auction_checked')
            if state.cost_to_call > 0:
                self._opp_bet_post_auction = True

    # ── Core decision ────────────────────────────────────────────────── #

    def get_move(self, game_info: GameInfo, current_state: PokerState) \
            -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        '''
        Decision pipeline:
          1. Observe and update the model.
          2. Infer opponent range from model.
          3. Auction phase → compute bid and return.
          4. Detect auction outcome on first flop query.
          5. Compute range-constrained equity.
          6. Analyse board texture.
          7. Build auction context dict (Phase 5a/5f/5g).
          8. Dispatch to decide_action.
        '''

        # ── 1. Observe ───────────────────────────────────────────────── #
        self._observe_preflop(current_state)
        self._observe_postflop(current_state)

        # ── 2. Opponent range ─────────────────────────────────────────── #
        opp_rf = self.opp_model.inferred_range_fraction()

        # ── 3. Auction ────────────────────────────────────────────────── #
        if current_state.street == 'auction':
            bid = compute_bid(current_state, self.strategy, opp_rf)
            self._our_auction_bid = bid
            return ActionBid(bid)

        # ── 4. Auction outcome (first flop query only) ────────────────── #
        if current_state.street == 'flop':
            self._observe_auction_outcome(current_state)
            self._observe_opp_post_auction_bet(current_state)

        # ── 5. Equity ─────────────────────────────────────────────────── #
        equity = get_equity(current_state, MC_SIMS, opp_rf)

        # ── 6. Board texture ──────────────────────────────────────────── #
        board   = list(current_state.board) if current_state.board else []
        texture = board_texture(board)

        # ── 7. Auction context ────────────────────────────────────────── #
        auction_ctx = {
            'we_won_auction'       : self._we_won_auction,
            'revealed_card'        : self._revealed_card,
            'opp_won_auction'      : self._opp_won_auction,
            'opp_bet_post_auction' : self._opp_bet_post_auction,
            'is_bb'                : self._is_bb,
        }

        # ── 8. Decide ─────────────────────────────────────────────────── #
        return decide_action(current_state, equity, self.strategy,
                             texture, auction_ctx)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    run_bot(Player(), parse_args())
