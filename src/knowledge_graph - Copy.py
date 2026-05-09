"""
knowledge_graph.py — 3-layer spreading-activation graph for terrain penalty.
CS F407 Project Assignment II

Layer 1 (source)   : Fragile, Valuable, Biohazard, Heavy
Layer 2 (property) : Hard, Wet, Slippery, Dirty, Visible,
                     Contaminated, Soft, Unstable
Layer 3 (terrain)  : Pavement, Grass, Dirt

Rules
-----
* Edges run L1→L2 and L2→L3 ONLY.
* All penalty scores are derived solely from edge-weight accumulation.
* No if/else chains or hard-coded penalty values anywhere.
* Final scores are capped at 1.0.

Design rationale for edge weights
-----------------------------------
Fragile  → Soft(0.9), Hard(0.3)  [prefers soft landing]
Fragile  → Unstable(0.1)          [unstable is risky for fragile]
Valuable → Visible(0.8), Dirty(0.2), Wet(0.2)
Biohazard→ Contaminated(0.9), Wet(0.6), Dirty(0.6)
Heavy    → Unstable(0.8), Soft(0.3), Hard(0.2)

Soft         → Grass(0.1), Dirt(0.3), Pavement(0.8)  [hard penalty for soft items]
Hard         → Pavement(0.2), Dirt(0.5), Grass(0.7)
Wet          → Grass(0.3), Dirt(0.6), Pavement(0.5)
Slippery     → Pavement(0.7), Grass(0.2), Dirt(0.4)
Dirty        → Dirt(0.1), Grass(0.3), Pavement(0.5)
Visible      → Pavement(0.2), Grass(0.3), Dirt(0.6)
Contaminated → Dirt(0.2), Grass(0.5), Pavement(0.8)
Unstable     → Dirt(0.6), Grass(0.4), Pavement(0.3)

Example verification: active=[Fragile, Valuable]
  Soft activated with 0.9 (from Fragile)
  Soft→Grass=0.1*0.9=0.09, Soft→Dirt=0.3*0.9=0.27, Soft→Pavement=0.8*0.9=0.72
  Hard activated with 0.3
  Hard→Grass=0.7*0.3=0.21, Hard→Dirt=0.5*0.3=0.15, Hard→Pavement=0.2*0.3=0.06
  Visible activated with 0.8 (from Valuable)
  Visible→Grass=0.3*0.8=0.24, Visible→Dirt=0.6*0.8=0.48, Visible→Pavement=0.2*0.8=0.16
  Dirty activated with 0.2
  Dirty→Grass=0.3*0.2=0.06, ...
  Total Grass  ≈ 0.09+0.21+0.24+... = 0.60
  Total Pavement ≈ 0.72+0.06+0.16+... = 0.94
  Total Dirt   ≈ 0.27+0.15+0.48+... = 0.90
  → Grass gets the lowest penalty  ✓ (soft landing for fragile+valuable cargo)
"""

# ── L1 → L2 edges: {source_node: {property_node: weight}} ────────────────────
SOURCE_TO_PROPERTY_EDGES = {
    "Fragile":   {"Soft": 0.9, "Hard": 0.3, "Unstable": 0.1},
    "Valuable":  {"Visible": 0.8, "Dirty": 0.2, "Wet": 0.2},
    "Biohazard": {"Contaminated": 0.9, "Wet": 0.6, "Dirty": 0.6},
    "Heavy":     {"Unstable": 0.8, "Soft": 0.3, "Hard": 0.2},
}

# ── L2 → L3 edges: {property_node: {terrain_node: weight}} ───────────────────
PROPERTY_TO_TERRAIN_EDGES = {
    "Soft":         {"Pavement": 0.8, "Dirt": 0.3, "Grass": 0.1},
    "Hard":         {"Grass": 0.7,   "Dirt": 0.5,  "Pavement": 0.2},
    "Wet":          {"Dirt": 0.6,    "Pavement": 0.5, "Grass": 0.3},
    "Slippery":     {"Pavement": 0.7, "Dirt": 0.4, "Grass": 0.2},
    "Dirty":        {"Pavement": 0.5, "Grass": 0.3, "Dirt": 0.1},
    "Visible":      {"Dirt": 0.6,    "Grass": 0.3, "Pavement": 0.2},
    "Contaminated": {"Pavement": 0.8, "Grass": 0.5, "Dirt": 0.2},
    "Unstable":     {"Dirt": 0.6,    "Grass": 0.4, "Pavement": 0.3},
}

# All terrain nodes
TERRAIN_NODES = ["Pavement", "Grass", "Dirt"]


def think(active_nodes: list) -> dict:
    """
    Spreading activation: accumulate penalty weights layer by layer.

    Parameters
    ----------
    active_nodes : list of active L1 source-node names
                   e.g. ["Fragile", "Valuable"]

    Returns
    -------
    dict mapping terrain name → penalty score in [0.0, 1.0]
    e.g. {"Pavement": 0.94, "Grass": 0.60, "Dirt": 0.90}
    """
    # Step 1 — Activate L2 property nodes via L1→L2 edges
    # property_activation[p] = sum of weights from all active source nodes
    property_activation = {prop: 0.0 for prop in PROPERTY_TO_TERRAIN_EDGES}

    for source in active_nodes:
        l1_edges = SOURCE_TO_PROPERTY_EDGES.get(source, {})
        for prop, weight in l1_edges.items():
            property_activation[prop] = property_activation.get(prop, 0.0) + weight

    # Step 2 — Propagate to L3 terrain nodes via L2→L3 edges
    terrain_score = {terrain: 0.0 for terrain in TERRAIN_NODES}

    for prop, activation in property_activation.items():
        l2_edges = PROPERTY_TO_TERRAIN_EDGES.get(prop, {})
        for terrain, weight in l2_edges.items():
            terrain_score[terrain] = terrain_score.get(terrain, 0.0) + activation * weight

    # Step 3 — Cap scores at 1.0
    terrain_score = {t: min(s, 1.0) for t, s in terrain_score.items()}

    return terrain_score
