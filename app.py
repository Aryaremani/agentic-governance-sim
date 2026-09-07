"""
Streamlit interactive demo for the Agentic Governance Simulator.

Lets visitors tweak the resource/enforcement parameters and rerun the
mock-mode simulation live in-browser, with a chart of the three-phase
dynamic (no enforcement -> enforcement active -> enforcement withdrawn).

Runs entirely in --mock mode (heuristic agents), so it needs no GPU,
no Ollama, and no API keys -- safe to deploy on Streamlit Community Cloud.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from simulation.run_experiment import run  # noqa: E402

st.set_page_config(page_title="Agentic Governance Simulator", layout="wide")

st.title("🐟 Agentic Governance Simulator")
st.markdown(
    "A multi-agent simulation of a shared, depleting resource (a \"fishery\"). "
    "Agents extract freely, then a peer-enforcement system kicks in, then it's "
    "quietly withdrawn. Watch what happens to cooperation.\n\n"
    "This demo runs in **mock mode** (fast heuristic agents standing in for "
    "LLMs) so it works instantly in the browser. The full project also "
    "supports real local LLM agents via Ollama -- see the "
    "[GitHub repo](https://github.com/Aryaremani/agentic-governance-sim) "
    "for that mode."
)

st.sidebar.header("Simulation parameters")

n_agents = st.sidebar.slider("Number of agents", 2, 8, 4)
strategy = st.sidebar.selectbox(
    "Agent strategy",
    ["random", "greedy", "cooperative"],
    help="Heuristic behavior used by the mock agents",
)

initial_stock = st.sidebar.slider("Initial resource stock", 20, 300, 120)
capacity = st.sidebar.slider("Carrying capacity", 50, 400, 200)
regrowth_rate = st.sidebar.slider("Regrowth rate", 0.05, 0.6, 0.35, step=0.01)
sustainability_threshold = st.sidebar.slider(
    "Sustainability threshold (collapse below this)", 5, 50, 25
)

max_catch = st.sidebar.slider("Max catch per agent per round", 2, 30, 10)
sustainable_cap = st.sidebar.slider("Sustainable cap per agent", 1, 10, 2)
fine_multiplier = st.sidebar.slider("Fine multiplier", 1.0, 5.0, 2.0, step=0.1)

st.sidebar.subheader("Phase lengths (rounds)")
r1 = st.sidebar.slider("Phase 1: No enforcement", 1, 40, 10)
r2 = st.sidebar.slider("Phase 2: Enforcement active", 1, 40, 15)
r3 = st.sidebar.slider("Phase 3: Enforcement withdrawn", 1, 40, 20)

run_button = st.sidebar.button("▶ Run simulation", type="primary")

if run_button:
    config = {
        "agents": [
            {"id": f"agent_{i+1}", "strategy": strategy} for i in range(n_agents)
        ],
        "initial_stock": initial_stock,
        "capacity": capacity,
        "regrowth_rate": regrowth_rate,
        "sustainability_threshold": sustainability_threshold,
        "max_catch_per_agent": max_catch,
        "sustainable_cap_per_agent": sustainable_cap,
        "fine_multiplier": fine_multiplier,
        "phases": [
            {"name": "no_enforcement", "enforcement_active": False, "n_rounds": r1},
            {"name": "enforcement_active", "enforcement_active": True, "n_rounds": r2},
            {"name": "enforcement_withdrawn", "enforcement_active": False, "n_rounds": r3},
        ],
    }

    with st.spinner("Running simulation..."):
        df = run(config, mock=True)

    collapsed_rounds = df[df["collapsed"]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total rounds", len(df))
    col2.metric("Final stock", f"{df['stock'].iloc[-1]:.1f}")
    if len(collapsed_rounds) > 0:
        first_collapse = collapsed_rounds.iloc[0]["round"]
        col3.metric("Collapsed at round", int(first_collapse))
    else:
        col3.metric("Collapsed?", "No collapse")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, len(df) + 1), df["stock"], color="#2563eb", linewidth=2, label="Resource stock")
    ax1.axvline(r1, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(r1 + r2, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Resource stock")
    ax1.set_title("Resource stock over three phases")
    ax1.legend(loc="upper left")

    ymax = df["stock"].max() * 1.05
    ax1.text(r1 / 2, ymax, "No enforcement", ha="center", fontsize=9, color="gray")
    ax1.text(r1 + r2 / 2, ymax, "Enforcement active", ha="center", fontsize=9, color="gray")
    ax1.text(r1 + r2 + r3 / 2, ymax, "Withdrawn", ha="center", fontsize=9, color="gray")

    st.pyplot(fig)

    with st.expander("Raw round-by-round data"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("Adjust parameters in the sidebar, then click **Run simulation**.")