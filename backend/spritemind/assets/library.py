"""PixelScript Asset Catalog and Procedural Definition Library for SpriteMind Stage 5.

Provides game-ready PixelScript programs, attachment sockets, hitboxes,
and metadata across Characters, Weapons, World Assets, and Effects/UI.
"""
from typing import Dict, Any, List, Optional, Tuple
from spritemind.pixelscript.ast import PixelScriptProgram, PixelAction, ActionType
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.utils.palettes import PALETTES


ASSET_CATEGORIES = [
    "characters",
    "weapons",
    "equipment",
    "world",
    "effects",
    "ui",
]

# Universal canonical sockets
CHARACTER_SOCKETS = ["hand_r", "hand_l", "head", "back", "torso_center"]
WEAPON_SOCKETS = ["grip", "muzzle", "optic", "barrel", "magazine"]


class AssetDefinition:
    """Structured container for an individual game-ready asset definition."""

    def __init__(
        self,
        asset_id: str,
        name: str,
        category: str,
        sub_category: str,
        width: int = 32,
        height: int = 32,
        script_text: str = "",
        sockets: Optional[Dict[str, Tuple[int, int]]] = None,
        hitboxes: Optional[Dict[str, List[int]]] = None,
        pivot: Tuple[int, int] = (16, 28),
        palette: str = "pico8",
        layer_order: int = 0,
        tags: Optional[List[str]] = None,
        description: str = "",
    ):
        self.asset_id = asset_id
        self.name = name
        self.category = category
        self.sub_category = sub_category
        self.width = width
        self.height = height
        self.script_text = script_text.strip()
        self.sockets = sockets or {}
        self.hitboxes = hitboxes or {}
        self.pivot = pivot
        self.palette = palette
        self.layer_order = layer_order
        self.tags = tags or []
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "sub_category": self.sub_category,
            "width": self.width,
            "height": self.height,
            "script_text": self.script_text,
            "sockets": {k: list(v) for k, v in self.sockets.items()},
            "hitboxes": self.hitboxes,
            "pivot": list(self.pivot),
            "palette": self.palette,
            "layer_order": self.layer_order,
            "tags": self.tags,
            "description": self.description,
        }


class AssetLibrary:
    """Catalog of canonical pixel-art game assets with PixelScript generators."""

    _registry: Dict[str, AssetDefinition] = {}

    @classmethod
    def register(cls, asset: AssetDefinition) -> None:
        cls._registry[asset.asset_id] = asset

    @classmethod
    def get(cls, asset_id: str) -> Optional[AssetDefinition]:
        cls._ensure_populated()
        return cls._registry.get(asset_id)

    @classmethod
    def list_all(cls, category: Optional[str] = None) -> List[AssetDefinition]:
        cls._ensure_populated()
        if category:
            return [a for a in cls._registry.values() if a.category == category]
        return list(cls._registry.values())

    @classmethod
    def _ensure_populated(cls) -> None:
        if cls._registry:
            return

        # -------------------------------------------------------------
        # 1. WEAPONS & EQUIPMENT
        # -------------------------------------------------------------
        # Pump Shotgun
        shotgun_script = """
LAYER 1 equipment 1.0
PART weapon none 0 0
RECT 2 6 12 3 5 1
RECT 14 6 10 2 0 1
LINE 8 9 6 13 4
RECT 16 8 4 2 2 1
LINE 2 6 24 6 0
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="weapon_pump_shotgun",
            name="12-Gauge Pump Action Shotgun",
            category="weapons",
            sub_category="shotgun",
            width=32,
            height=32,
            script_text=shotgun_script,
            sockets={"grip": (7, 10), "muzzle": (24, 7), "optic": (12, 5), "barrel": (24, 7)},
            hitboxes={"weapon_bounds": [2, 5, 22, 8]},
            pivot=(7, 10),
            layer_order=3,
            tags=["weapon", "firearm", "deadzone", "shotgun"],
            description="Heavy pump-action shotgun with wooden stock and steel heat shield.",
        ))

        # Tactical Pistol 9mm
        pistol_script = """
LAYER 1 equipment 1.0
PART weapon none 0 0
RECT 10 10 8 3 0 1
RECT 8 13 4 5 1 1
LINE 10 11 18 11 6
PIXEL 17 9 7
PIXEL 11 9 7
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="weapon_tactical_pistol",
            name="9mm Tactical Pistol",
            category="weapons",
            sub_category="pistol",
            width=32,
            height=32,
            script_text=pistol_script,
            sockets={"grip": (10, 14), "muzzle": (18, 11), "optic": (13, 9), "barrel": (18, 11)},
            hitboxes={"weapon_bounds": [8, 9, 10, 9]},
            pivot=(10, 14),
            layer_order=3,
            tags=["weapon", "pistol", "sidearm"],
            description="Semi-automatic 9mm handgun with tactical iron sights.",
        ))

        # Assault Rifle
        rifle_script = """
LAYER 1 equipment 1.0
PART weapon none 0 0
RECT 4 8 18 3 1 1
RECT 2 9 5 4 4 1
RECT 12 11 4 6 0 1
LINE 22 9 26 9 6
RECT 8 11 3 4 0 1
LINE 6 7 18 7 0
PIXEL 20 8 7
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="weapon_assault_rifle",
            name="Assault Rifle AR-24",
            category="weapons",
            sub_category="rifle",
            width=32,
            height=32,
            script_text=rifle_script,
            sockets={"grip": (9, 12), "muzzle": (26, 9), "optic": (12, 6), "barrel": (26, 9), "magazine": (14, 16)},
            hitboxes={"weapon_bounds": [2, 6, 24, 11]},
            pivot=(9, 12),
            layer_order=3,
            tags=["weapon", "rifle", "automatic"],
            description="Standard military issue selective-fire automatic carbine.",
        ))

        # Heavy Minigun
        minigun_script = """
LAYER 1 equipment 1.0
PART weapon none 0 0
RECT 4 7 10 7 1 1
RECT 14 8 12 2 5 1
RECT 14 11 12 2 5 1
LINE 26 7 26 14 0
RECT 6 14 4 6 0 1
RECT 2 6 4 4 2 1
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="weapon_heavy_minigun",
            name="Vulcan Rotary Minigun",
            category="weapons",
            sub_category="heavy",
            width=32,
            height=32,
            script_text=minigun_script,
            sockets={"grip": (7, 16), "muzzle": (26, 10), "optic": (10, 5), "barrel": (26, 10)},
            hitboxes={"weapon_bounds": [2, 6, 24, 14]},
            pivot=(7, 16),
            layer_order=3,
            tags=["weapon", "heavy", "rotary"],
            description="Heavy six-barrel rotating autocannon for suppressive fire.",
        ))

        # Combat Knife
        knife_script = """
LAYER 1 equipment 1.0
PART weapon none 0 0
RECT 10 14 3 6 4 1
RECT 9 12 5 2 0 1
LINE 11 12 11 4 7
LINE 12 12 12 6 6
PIXEL 11 3 7
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="weapon_combat_knife",
            name="Tactical Combat Knife",
            category="weapons",
            sub_category="melee",
            width=32,
            height=32,
            script_text=knife_script,
            sockets={"grip": (11, 16), "muzzle": (11, 3)},
            hitboxes={"weapon_bounds": [9, 3, 5, 17]},
            pivot=(11, 16),
            layer_order=3,
            tags=["weapon", "melee", "knife"],
            description="Serrated steel survival dagger with rubberized grip.",
        ))

        # Tactical Helmet
        helmet_script = """
LAYER 2 armor 1.0
PART helmet none 0 0
RECT 10 4 12 8 1 1
LINE 9 8 23 8 0
RECT 12 9 8 3 12 1
LINE 10 4 22 4 0
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="armor_combat_helmet",
            name="Kevlar Combat Helmet",
            category="equipment",
            sub_category="helmet",
            width=32,
            height=32,
            script_text=helmet_script,
            sockets={"head": (16, 10)},
            pivot=(16, 10),
            layer_order=2,
            tags=["equipment", "armor", "helmet"],
            description="Ballistic Kevlar helmet with anti-fragmentation visor.",
        ))

        # Military Backpack
        backpack_script = """
LAYER -1 back 1.0
PART backpack none 0 0
RECT 6 12 7 12 4 1
RECT 7 14 5 4 2 1
RECT 7 20 5 3 0 1
LINE 5 12 5 24 0
LINE 13 12 13 24 0
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="equipment_military_backpack",
            name="Rucksack Supply Backpack",
            category="equipment",
            sub_category="backpack",
            width=32,
            height=32,
            script_text=backpack_script,
            sockets={"back": (13, 16)},
            pivot=(13, 16),
            layer_order=-1,
            tags=["equipment", "backpack", "storage"],
            description="Field rucksack for ammunition and survival rations.",
        ))

        # Silencer Attachment
        silencer_script = """
LAYER 4 attachment 1.0
PART silencer none 0 0
RECT 0 6 8 4 0 1
LINE 1 7 7 7 1
LINE 1 8 7 8 5
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="attachment_suppressor",
            name="Tactical Sound Suppressor",
            category="equipment",
            sub_category="attachment",
            width=16,
            height=16,
            script_text=silencer_script,
            sockets={"barrel": (0, 8)},
            pivot=(0, 8),
            layer_order=4,
            tags=["attachment", "stealth", "barrel"],
            description="Baffled steel barrel suppressor reducing acoustic signature.",
        ))

        # -------------------------------------------------------------
        # 2. CHARACTERS & ENEMIES
        # -------------------------------------------------------------
        # Base Survivor (Player Soldier)
        soldier_script = """
LAYER 0 body 1.0
PART torso none 16 16
RECT 12 14 8 10 3 1
END PART
PART head torso 16 8
RECT 13 6 6 8 15 1
PIXEL 15 9 0
PIXEL 17 9 0
LINE 13 6 19 6 4
END PART
PART arm_l torso 10 16
RECT 9 14 3 8 3 1
PIXEL 10 22 15
END PART
PART arm_r torso 22 16
RECT 20 14 3 8 3 1
PIXEL 21 22 15
END PART
PART leg_l torso 13 24
RECT 12 24 3 7 1 1
RECT 11 30 4 2 0 1
END PART
PART leg_r torso 19 24
RECT 17 24 3 7 1 1
RECT 17 30 4 2 0 1
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="character_player_soldier",
            name="Special Forces Operative",
            category="characters",
            sub_category="player",
            width=32,
            height=32,
            script_text=soldier_script,
            sockets={"hand_r": (21, 22), "hand_l": (10, 22), "head": (16, 7), "back": (16, 16), "torso_center": (16, 18)},
            hitboxes={"body": [10, 6, 12, 25], "head": [13, 6, 6, 8]},
            pivot=(16, 31),
            layer_order=0,
            tags=["character", "player", "deadzone", "soldier"],
            description="Human survival soldier in tactical fatigue gear.",
        ))

        # Zombie Walker
        zombie_script = """
LAYER 0 body 1.0
PART torso none 16 16
RECT 12 14 8 10 11 1
LINE 14 16 16 18 8
LINE 13 20 17 21 8
END PART
PART head torso 16 8
RECT 13 6 6 8 3 1
PIXEL 14 9 8
PIXEL 17 9 8
PIXEL 15 12 0
END PART
PART arm_l torso 9 15
RECT 8 13 3 8 3 1
PIXEL 9 21 11
END PART
PART arm_r torso 22 15
RECT 21 12 3 9 3 1
PIXEL 22 21 11
END PART
PART leg_l torso 13 24
RECT 12 24 3 7 1 1
PIXEL 13 27 8
RECT 11 30 4 2 0 1
END PART
PART leg_r torso 19 24
RECT 17 24 3 7 1 1
RECT 17 30 4 2 0 1
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="character_zombie_walker",
            name="Infected Zombie Walker",
            category="characters",
            sub_category="zombie",
            width=32,
            height=32,
            script_text=zombie_script,
            sockets={"hand_r": (22, 20), "hand_l": (9, 21), "head": (16, 7), "back": (16, 16), "torso_center": (16, 18)},
            hitboxes={"body": [9, 6, 14, 25], "head": [13, 6, 6, 8]},
            pivot=(16, 31),
            layer_order=0,
            tags=["character", "enemy", "zombie", "undead", "deadzone"],
            description="Lethal reanimated infected corpse with decaying flesh.",
        ))

        # Combat Sentry Robot
        robot_script = """
LAYER 0 body 1.0
PART torso none 16 16
RECT 11 13 10 10 5 1
RECT 13 15 6 4 12 1
END PART
PART head torso 16 7
RECT 12 6 8 7 6 1
LINE 13 9 19 9 8
LINE 16 3 16 6 0
PIXEL 16 2 8
END PART
PART arm_l torso 8 16
RECT 8 14 3 8 5 1
RECT 7 21 4 3 0 1
END PART
PART arm_r torso 24 16
RECT 21 14 3 8 5 1
RECT 21 21 4 3 0 1
END PART
PART leg_l torso 13 24
RECT 12 23 3 8 0 1
RECT 11 30 4 2 6 1
END PART
PART leg_r torso 19 24
RECT 17 23 3 8 0 1
RECT 17 30 4 2 6 1
END PART
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="character_combat_robot",
            name="Autonomous Sentry Android",
            category="characters",
            sub_category="robot",
            width=32,
            height=32,
            script_text=robot_script,
            sockets={"hand_r": (22, 22), "hand_l": (8, 22), "head": (16, 5), "back": (16, 16), "torso_center": (16, 17)},
            hitboxes={"body": [8, 5, 16, 26], "core": [13, 15, 6, 4]},
            pivot=(16, 31),
            layer_order=0,
            tags=["character", "enemy", "robot", "mechanical"],
            description="Heavy armored steel combat chassis with infrared optical sensors.",
        ))

        # -------------------------------------------------------------
        # 3. WORLD ASSETS (PICKUPS, CRATES, OBSTACLES)
        # -------------------------------------------------------------
        # Military Medkit Pickup
        medkit_script = """
LAYER 0 prop 1.0
RECT 8 14 16 12 7 1
RECT 9 15 14 10 0 1
RECT 10 16 12 8 7 1
RECT 14 18 4 4 8 1
RECT 12 19 8 2 8 1
LINE 13 13 19 13 0
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="world_medkit_large",
            name="First Aid Trauma Kit",
            category="world",
            sub_category="pickup",
            width=32,
            height=32,
            script_text=medkit_script,
            hitboxes={"trigger": [8, 13, 16, 13]},
            pivot=(16, 26),
            tags=["world", "pickup", "health", "deadzone"],
            description="Restores 50 Hit Points and cures infection wounds.",
        ))

        # Heavy Ammo Box
        ammo_box_script = """
LAYER 0 prop 1.0
RECT 7 15 18 11 3 1
RECT 8 16 16 9 11 1
LINE 7 18 25 18 0
LINE 13 19 19 19 10
RECT 14 20 4 3 10 1
LINE 11 14 21 14 0
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="world_ammo_box_heavy",
            name="Military Munitions Case",
            category="world",
            sub_category="pickup",
            width=32,
            height=32,
            script_text=ammo_box_script,
            hitboxes={"trigger": [7, 14, 18, 12]},
            pivot=(16, 26),
            tags=["world", "pickup", "ammo", "deadzone"],
            description="Refills 12-gauge shotgun shells and 9mm magazines.",
        ))

        # Explosive Red Barrel
        barrel_script = """
LAYER 0 prop 1.0
RECT 9 10 14 18 8 1
RECT 10 11 12 16 2 1
LINE 9 14 23 14 0
LINE 9 22 23 22 0
RECT 14 17 4 4 7 1
PIXEL 15 18 8
PIXEL 16 19 8
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="world_explosive_barrel",
            name="Combustible Fuel Drum",
            category="world",
            sub_category="obstacle",
            width=32,
            height=32,
            script_text=barrel_script,
            hitboxes={"solid": [9, 10, 14, 18], "explosion_radius": [0, 0, 32, 32]},
            pivot=(16, 28),
            tags=["world", "obstacle", "destructible", "explosive", "deadzone"],
            description="Volatile pressurized red drum. Detonates on projectile impact.",
        ))

        # Reinforced Metal Blast Door
        door_script = """
LAYER 0 prop 1.0
RECT 4 4 24 26 0 1
RECT 6 6 20 22 5 1
LINE 16 6 16 28 0
RECT 8 8 6 6 1 1
RECT 18 8 6 6 1 1
RECT 8 18 6 6 1 1
RECT 18 18 6 6 1 1
PIXEL 14 16 10
PIXEL 18 16 8
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="world_blast_door",
            name="Reinforced Security Door",
            category="world",
            sub_category="door",
            width=32,
            height=32,
            script_text=door_script,
            hitboxes={"solid": [4, 4, 24, 26]},
            pivot=(16, 30),
            tags=["world", "structure", "door", "environment"],
            description="High-security blast vault bulkhead.",
        ))

        # -------------------------------------------------------------
        # 4. EFFECTS & UI
        # -------------------------------------------------------------
        # Muzzle Flash (Shotgun blast)
        muzzle_flash_script = """
LAYER 4 fx 1.0
PIXEL 2 15 7
LINE 4 14 4 17 10
RECT 6 13 4 5 10 1
RECT 8 14 6 3 7 1
LINE 14 12 18 12 9
LINE 14 18 18 18 9
PIXEL 16 15 7
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="fx_muzzle_flash_shotgun",
            name="12G Kinetic Muzzle Flash",
            category="effects",
            sub_category="muzzle_flash",
            width=32,
            height=32,
            script_text=muzzle_flash_script,
            sockets={"muzzle": (2, 15)},
            pivot=(2, 15),
            layer_order=4,
            tags=["fx", "muzzle_flash", "combat", "particles"],
            description="High-intensity incandescent powder ignition flame.",
        ))

        # Bullet Impact Sparks
        sparks_script = """
LAYER 4 fx 1.0
PIXEL 16 16 7
PIXEL 14 13 10
PIXEL 19 12 9
PIXEL 13 18 9
PIXEL 20 19 10
PIXEL 11 11 8
PIXEL 22 10 8
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="fx_bullet_impact_sparks",
            name="Ricochet Impact Sparks",
            category="effects",
            sub_category="impact",
            width=32,
            height=32,
            script_text=sparks_script,
            pivot=(16, 16),
            layer_order=4,
            tags=["fx", "impact", "sparks"],
            description="High-velocity particulate ricochet burst.",
        ))

        # Blood Splatter FX
        blood_script = """
LAYER 4 fx 1.0
RECT 13 14 6 5 8 1
LINE 11 16 21 16 2
PIXEL 10 12 8
PIXEL 18 10 2
PIXEL 22 14 8
PIXEL 12 21 2
PIXEL 17 22 8
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="fx_blood_splatter",
            name="Organic Blood Splatter",
            category="effects",
            sub_category="hit",
            width=32,
            height=32,
            script_text=blood_script,
            pivot=(16, 16),
            layer_order=4,
            tags=["fx", "blood", "damage"],
            description="Visceral arterial blood droplet dispersion.",
        ))

        # UI Kill Skull Icon
        icon_skull_script = """
LAYER 0 ui 1.0
RECT 8 6 16 12 7 1
RECT 10 18 12 6 7 1
RECT 10 9 4 5 0 1
RECT 18 9 4 5 0 1
RECT 14 14 4 3 0 1
LINE 12 21 12 24 0
LINE 15 21 15 24 0
LINE 18 21 18 24 0
END PROGRAM
"""
        cls.register(AssetDefinition(
            asset_id="ui_icon_skull_kill",
            name="Skull Elimination Badge",
            category="ui",
            sub_category="icon",
            width=32,
            height=32,
            script_text=icon_skull_script,
            pivot=(16, 16),
            tags=["ui", "icon", "hud"],
            description="HUD killfeed confirmation indicator.",
        ))
