"""Action catalog: reusable animation building blocks for prompt-driven sets."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    anim: str
    frames: int = 6
    fps: int = 10
    loop: bool = False
    effects: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "anim": self.anim,
            "frames": self.frames,
            "fps": self.fps,
            "loop": self.loop,
            "effects": list(self.effects),
        }


@dataclass
class ActionPlan:
    prompt: str
    category: str
    style: str
    palette: str
    width: int
    height: int
    archetype: str
    element: str
    weapon: str
    proportion: str = "standard"
    accessory: str = "none"
    actions: List[ActionSpec] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "style": self.style,
            "palette": self.palette,
            "width": self.width,
            "height": self.height,
            "archetype": self.archetype,
            "element": self.element,
            "weapon": self.weapon,
            "proportion": self.proportion,
            "accessory": self.accessory,
            "actions": [a.to_dict() for a in self.actions],
            "rationale": list(self.rationale),
        }


LOCOMOTION = [
    ActionSpec("idle", "Idle", "idle", frames=4, fps=6, loop=True),
    ActionSpec("walk", "Walk", "walk", frames=6, fps=8, loop=True),
    ActionSpec("run", "Run", "run", frames=6, fps=12, loop=True),
    ActionSpec("dash", "Dash", "dash", frames=5, fps=14, effects=("dash_trail",)),
]

MELEE = [
    ActionSpec("attack", "Attack", "melee_attack", frames=6, fps=12, effects=("slash_arc",)),
    ActionSpec("attack_heavy", "Heavy Attack", "melee_attack", frames=8, fps=10, effects=("slash_arc", "impact")),
]

ARCANE = [
    ActionSpec("cast", "Cast", "cast", frames=7, fps=9, effects=("orb_charge", "projectile")),
    ActionSpec("cast_burst", "Spell Burst", "special_ability", frames=6, fps=9, effects=("aura", "impact")),
]

RANGED = [ActionSpec("shoot", "Shoot", "ranged_attack", frames=6, fps=12, effects=("projectile",))]

DEFENSE = [
    ActionSpec("block", "Block", "blocking", frames=4, fps=8),
    ActionSpec("hurt", "Hurt", "hit_reaction", frames=4, fps=10, effects=("impact",)),
]

AIRBORNE = [ActionSpec("jump", "Jump", "jump", frames=6, fps=10)]
FALL = [ActionSpec("fall", "Fall", "fall", frames=5, fps=8, loop=True)]
DEATH = [ActionSpec("death", "Death", "death", frames=7, fps=8)]

ARCHETYPE_ACTIONS: Dict[str, List[ActionSpec]] = {
    "warrior": LOCOMOTION + [MELEE[0], MELEE[1]] + DEFENSE[:1] + [DEFENSE[1]] + DEATH,
    "ranger": LOCOMOTION + RANGED + [MELEE[0]] + [DEFENSE[1], AIRBORNE[0]] + DEATH,
    "arcane": LOCOMOTION + ARCANE + [DEFENSE[1]] + DEATH,
    "rogue": LOCOMOTION + [MELEE[0]] + [AIRBORNE[0]] + [DEFENSE[1]] + DEATH,
    "brute": LOCOMOTION[:-1] + MELEE + DEFENSE + DEATH,
    "critter": LOCOMOTION + [AIRBORNE[0], MELEE[0]] + [DEFENSE[1]] + DEATH,
}

CATEGORY_ARCHETYPE = {
    "knight": "warrior",
    "skeleton": "warrior",
    "goblin": "rogue",
    "zombie": "brute",
    "wizard": "arcane",
    "robot": "ranger",
    "rogue": "rogue",
    "slime": "critter",
    "animal": "critter",
    "fantasy_creature": "brute",
}

ELEMENT_KEYWORDS = {
    "fire": ("fire", "flame", "flaming", "burn", "burning", "ember", "lava", "magma", "inferno", "blaze"),
    "ice": ("ice", "icy", "frost", "frozen", "snow", "glacial", "blizzard"),
    "lightning": ("lightning", "electric", "electricity", "thunder", "shock", "storm"),
    "holy": ("holy", "divine", "sacred", "light", "angel", "radiant", "cleric", "paladin"),
    "shadow": ("shadow", "dark", "void", "necro", "cursed", "demon", "abyss"),
    "poison": ("poison", "venom", "toxic", "acid", "plague", "swamp"),
    "arcane": ("arcane", "magic", "magical", "spell", "enchant", "rune", "mystic", "psychic"),
}

ELEMENT_DEFAULT_BY_CATEGORY = {
    "wizard": "arcane",
    "skeleton": "shadow",
    "zombie": "poison",
    "fantasy_creature": "fire",
    "robot": "lightning",
    "slime": "poison",
}

ARCHETYPE_KEYWORDS = {
    "arcane": ("mage", "wizard", "sorcerer", "witch", "warlock", "spell", "spellcaster", "priest", "cleric", "druid", "shaman", "necromancer"),
    "ranger": ("archer", "ranger", "hunter", "marksman", "sniper", "gunner", "gunslinger", "sharpshooter", "crossbow", "bow", "gun"),
    "rogue": ("rogue", "assassin", "thief", "ninja", "scout", "dagger", "shadowblade"),
    "brute": ("brute", "berserker", "barbarian", "ogre", "golem", "troll", "tank", "heavy", "giant"),
    "warrior": ("knight", "warrior", "paladin", "soldier", "guard", "fighter", "swordsman", "samurai", "hero"),
}

WEAPON_KEYWORDS = [
    ("staff", ("staff", "wand", "scepter", "sceptre", "rod", "orb")),
    ("bow", ("bow", "crossbow", "arrow", "arrowhead", "longbow")),
    ("axe", ("axe", "halberd", "cleaver")),
    ("dagger", ("dagger", "knife", "shiv", "stiletto")),
    ("shield", ("shield", "buckler", "aegis")),
    ("sword", ("sword", "blade", "katana", "greatsword", "longsword", "sabre", "saber")),
]

WEAPON_BY_ARCHETYPE = {"arcane": "staff", "ranger": "bow", "rogue": "dagger", "brute": "axe", "warrior": "sword", "critter": "none"}

ACTION_KEYWORDS = {
    "dash": ("dash", "dodgeroll", "dodge roll", "roll", "sprint", "blink", "charge forward"),
    "attack_heavy": ("heavy attack", "smash", "slam", "cleave", "charged attack", "combo finisher"),
    "block": ("block", "shield up", "parry", "guard", "defend"),
    "jump": ("jump", "leap", "hop", "vault"),
    "fall": ("fall", "falling", "descend"),
    "shoot": ("shoot", "fire an arrow", "ranged", "gun", "crossbow", "snipe"),
    "cast": ("cast", "spell", "magic attack", "fireball", "projectile", "summon", "hex", "bolt"),
}
