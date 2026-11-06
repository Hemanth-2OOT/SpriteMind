"""Tests for prompt-driven action sets."""
import base64

from spritemind.actions import ActionSetBuilder, ActionSetPlanner, build_effects
from spritemind.actions.effects import element_colors


def test_planner_infers_mage_cast_and_dash():
    plan = ActionSetPlanner.plan("a fire mage that casts fireballs and dashes")
    keys = [a.key for a in plan.actions]
    assert plan.category == "wizard"
    assert plan.archetype == "arcane"
    assert plan.element == "fire"
    assert plan.weapon == "staff"
    assert "cast" in keys and "dash" in keys


def test_planner_infers_melee_and_defense():
    plan = ActionSetPlanner.plan("dark fantasy knight with shield who blocks and smashes")
    keys = [a.key for a in plan.actions]
    assert plan.archetype == "warrior"
    assert plan.element == "shadow"
    assert plan.weapon == "shield"
    assert "block" in keys and "attack_heavy" in keys


def test_planner_defaults_are_stable():
    plan = ActionSetPlanner.plan("")
    assert plan.category == "knight"
    assert plan.actions[0].key == "idle"
    assert len(plan.actions) <= 12


def test_planner_respects_requested_actions():
    plan = ActionSetPlanner.plan("wizard", requested_actions=["cast", "dash"])
    assert [a.key for a in plan.actions] == ["idle", "cast", "dash"]


def test_element_colors_are_valid_palette_indices():
    for element in ("fire", "ice", "lightning", "holy", "shadow", "poison", "arcane"):
        glow, core, spark = element_colors("pico8", element)
        assert all(isinstance(i, int) and i > 0 for i in (glow, core, spark))


def test_effects_are_phase_gated():
    assert build_effects("projectile", 0.0) == ""
    assert build_effects("orb_charge", 0.1) == ""
    mid = build_effects("orb_charge", 0.6, element="fire")
    assert "PART fx_orb" in mid
    assert build_effects("dash_trail", 0.4) != ""


def test_builder_produces_coherent_set():
    result = ActionSetBuilder.build(
        prompt="ice paladin with shield who blocks",
        seed=3,
        max_actions=3,
        include_gif=False,
    )
    assert result["num_actions"] == 3
    assert result["plan"]["element"] == "ice"
    for action in result["actions"]:
        assert action["num_frames"] == len(action["frames"])
        for frame in action["frames"]:
            assert frame["validation"]["is_valid"] is True
            assert frame["image_base64"].startswith("data:image/png;base64,")
            assert base64.b64decode(frame["image_base64"].split(",", 1)[1])


def test_builder_attaches_effects_to_cast_frames():
    result = ActionSetBuilder.build(prompt="fire mage", seed=5, actions=["cast"], include_gif=False)
    cast = next(a for a in result["actions"] if a["key"] == "cast")
    assert cast["key"] == "cast"
    assert any("fx_orb" in f["script_text"] for f in cast["frames"])


def test_plan_endpoint_returns_actions():
    from fastapi.testclient import TestClient

    from spritemind.server.app import app

    client = TestClient(app)
    res = client.post("/api/action_set/plan", json={"prompt": "goblin with a spear"})
    assert res.status_code == 200
    plan = res.json()["plan"]
    assert plan["actions"][0]["key"] == "idle"
    assert any(a["key"] in ("attack", "attack_heavy") for a in plan["actions"])
