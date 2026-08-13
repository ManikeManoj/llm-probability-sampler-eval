# The final specification for the LLM probability sampler evaluation.

DECIMALS = 3
N_SAMPLES = 500_000
MC_SEED = 42
MC_RELIABLE_THRESHOLD = 1_000
ICL_N_EXAMPLES = 5
ICL_SEED = 0
LM_SCORING_METHOD = "single_token"


# ---------------------------------------------------------------------
# Model / protocol conditions
# ---------------------------------------------------------------------

SMALL_MODEL_PROTOCOLS = [
    {
        "id": "S1",
        "model_name": "Qwen/Qwen3-4B-Base",
        "protocol": "raw_direct",
    },
    {
        "id": "S2",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "protocol": "raw_direct",
    },
    {
        "id": "S3",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "protocol": "chat_direct",
    },
    {
        "id": "S4",
        "model_name": "google/gemma-4-E4B",
        "protocol": "raw_direct",
    },
    {
        "id": "S5",
        "model_name": "google/gemma-4-E4B-it",
        "protocol": "raw_direct",
    },
    {
        "id": "S6",
        "model_name": "google/gemma-4-E4B-it",
        "protocol": "chat_direct",
    },
]


MEDIUM_MODEL_PROTOCOLS = [
    {
        "id": "M1",
        "model_name": "Qwen/Qwen3-14B-Base",
        "protocol": "raw_direct",
    },
    {
        "id": "M2",
        "model_name": "Qwen/Qwen3-14B",
        "protocol": "raw_direct",
    },
    {
        "id": "M3",
        "model_name": "Qwen/Qwen3-14B",
        "protocol": "chat_direct",
    },
    {
        "id": "M4",
        "model_name": "google/gemma-4-12B",
        "protocol": "raw_direct",
    },
    {
        "id": "M5",
        "model_name": "google/gemma-4-12B-it",
        "protocol": "raw_direct",
    },
    {
        "id": "M6",
        "model_name": "google/gemma-4-12B-it",
        "protocol": "chat_direct",
    },
]


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

ALL_PROMPT_TYPES = [
    "short",
    "plain",
    "formal",
    "explanatory_1",
    "explanatory_2",
    "explanatory_3",
    "explanatory_4",
    "cot",
    "icl",
    "icl_random",
    "icl_cot",
]

MAIN_PROMPT_TYPES = [
    "plain",
    "explanatory_4",
]


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

BASELINE_PARAMETER_IDS = {
    "normal": "N1",
    "uniform": "U1",
    "exponential": "E1",
    "beta": "B2",
    "laplace": "L1",
}


# ---------------------------------------------------------------------
# Parameter + prefix grid
#
# Prefix groups:
#   root          -> beginning of number
#   sign          -> after "-"
#   integer       -> integer continuation / termination
#   dot           -> first fractional digit
#   fraction_d1   -> second fractional digit
#   fraction_d2   -> third fractional digit
#
# ROOT is stored textually here for readability and is converted to ""
# by the validator / manifest generator.
# ---------------------------------------------------------------------

PARAMETER_CONFIGS = [

    # ================================================================
    # NORMAL
    # ================================================================

    {
        "id": "N1",
        "distribution": "normal",
        "params": {"mean": 0.0, "std": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-1", "-0", "0", "1"],
            "dot": ["-1.", "-0.", "0.", "1."],
            "fraction_d1": ["-0.5", "-0.0", "0.0", "0.5"],
            "fraction_d2": ["-0.50", "-0.00", "0.00", "0.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "N2",
        "distribution": "normal",
        "params": {"mean": 0.0, "std": 0.5},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-1", "-0", "0", "1"],
            "dot": ["-1.", "-0.", "0.", "1."],
            "fraction_d1": ["-0.2", "-0.0", "0.0", "0.2"],
            "fraction_d2": ["-0.20", "-0.00", "0.00", "0.20"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "N3",
        "distribution": "normal",
        "params": {"mean": 0.0, "std": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-2", "-1", "-0", "0", "1", "2"],
            "dot": ["-2.", "-1.", "-0.", "0.", "1.", "2."],
            "fraction_d1": ["-0.1", "-0.0", "0.0", "0.1"],
            "fraction_d2": ["-0.10", "-0.00", "0.00", "0.10"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "N4",
        "distribution": "normal",
        "params": {"mean": 2.0, "std": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-0", "0", "1", "2", "3"],
            "dot": ["-0.", "0.", "1.", "2.", "3."],
            "fraction_d1": ["1.5", "2.0", "2.5"],
            "fraction_d2": ["1.50", "2.00", "2.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "N5",
        "distribution": "normal",
        "params": {"mean": -2.0, "std": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-3", "-2", "-1", "-0", "0"],
            "dot": ["-3.", "-2.", "-1.", "-0.", "0."],
            "fraction_d1": ["-2.5", "-2.0", "-1.5"],
            "fraction_d2": ["-2.50", "-2.00", "-1.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "N6",
        "distribution": "normal",
        "params": {"mean": 0.0, "std": 10.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-10", "-1", "-0", "0", "1", "10"],
            "dot": ["-10.", "-1.", "-0.", "0.", "1.", "10."],
            "fraction_d1": ["-1.0", "-0.0", "0.0", "1.0"],
            "fraction_d2": ["-1.00", "-0.00", "0.00", "1.00"],
        },
        "magnitude_prefixes": ["-1", "1"],
    },

    # ================================================================
    # UNIFORM
    # ================================================================

    {
        "id": "U1",
        "distribution": "uniform",
        "params": {"low": 0.0, "high": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "U2",
        "distribution": "uniform",
        "params": {"low": -1.0, "high": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-0", "0"],
            "dot": ["-0.", "0."],
            "fraction_d1": ["-0.5", "0.0", "0.5"],
            "fraction_d2": ["-0.50", "0.00", "0.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "U3",
        "distribution": "uniform",
        "params": {"low": 0.0, "high": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0", "1"],
            "dot": ["0.", "1."],
            "fraction_d1": ["0.5", "1.0", "1.5"],
            "fraction_d2": ["0.50", "1.00", "1.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "U4",
        "distribution": "uniform",
        "params": {"low": 0.0, "high": 100.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["1", "5", "10", "50"],
            "dot": ["1.", "5.", "10.", "50."],
            "fraction_d1": ["1.5", "50.5"],
            "fraction_d2": ["1.50", "50.50"],
        },
        "magnitude_prefixes": ["1", "5"],
    },

    # ================================================================
    # EXPONENTIAL
    # ================================================================

    {
        "id": "E1",
        "distribution": "exponential",
        "params": {"rate": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0", "1", "2"],
            "dot": ["0.", "1.", "2."],
            "fraction_d1": ["0.1", "0.5", "1.1"],
            "fraction_d2": ["0.10", "0.50", "1.10"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "E2",
        "distribution": "exponential",
        "params": {"rate": 0.5},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0", "1", "2"],
            "dot": ["0.", "1.", "2."],
            "fraction_d1": ["0.1", "0.5", "1.1"],
            "fraction_d2": ["0.10", "0.50", "1.10"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "E3",
        "distribution": "exponential",
        "params": {"rate": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0", "1", "2"],
            "dot": ["0.", "1.", "2."],
            "fraction_d1": ["0.1", "0.5", "1.1"],
            "fraction_d2": ["0.10", "0.50", "1.10"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "E4",
        "distribution": "exponential",
        "params": {"rate": 0.1},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0", "1", "2", "10", "20"],
            "dot": ["0.", "1.", "2.", "10.", "20."],
            "fraction_d1": ["0.1", "0.5", "1.1", "10.0"],
            "fraction_d2": ["0.10", "0.50", "1.10", "10.00"],
        },
        "magnitude_prefixes": ["1", "2"],
    },

    # ================================================================
    # BETA
    # SAME PREFIXES B1-B6 INTENTIONALLY
    # ================================================================

    {
        "id": "B1",
        "distribution": "beta",
        "params": {"alpha": 1.0, "beta": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "B2",
        "distribution": "beta",
        "params": {"alpha": 2.0, "beta": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "B3",
        "distribution": "beta",
        "params": {"alpha": 0.5, "beta": 0.5},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "B4",
        "distribution": "beta",
        "params": {"alpha": 2.0, "beta": 5.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "B5",
        "distribution": "beta",
        "params": {"alpha": 5.0, "beta": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "B6",
        "distribution": "beta",
        "params": {"alpha": 5.0, "beta": 5.0},
        "prefixes": {
            "root": ["ROOT"],
            "integer": ["0"],
            "dot": ["0."],
            "fraction_d1": ["0.2", "0.5", "0.8"],
            "fraction_d2": ["0.20", "0.50", "0.80"],
        },
        "magnitude_prefixes": [],
    },

    # ================================================================
    # LAPLACE
    # ================================================================

    {
        "id": "L1",
        "distribution": "laplace",
        "params": {"loc": 0.0, "scale": 1.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-1", "-0", "0", "1"],
            "dot": ["-1.", "-0.", "0.", "1."],
            "fraction_d1": ["-0.5", "-0.0", "0.0", "0.5"],
            "fraction_d2": ["-0.50", "-0.00", "0.00", "0.50"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "L2",
        "distribution": "laplace",
        "params": {"loc": 0.0, "scale": 0.5},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-1", "-0", "0", "1"],
            "dot": ["-1.", "-0.", "0.", "1."],
            "fraction_d1": ["-0.2", "-0.0", "0.0", "0.2"],
            "fraction_d2": ["-0.20", "-0.00", "0.00", "0.20"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "L3",
        "distribution": "laplace",
        "params": {"loc": 0.0, "scale": 2.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-2", "-1", "-0", "0", "1", "2"],
            "dot": ["-2.", "-1.", "-0.", "0.", "1.", "2."],
            "fraction_d1": ["-0.1", "-0.0", "0.0", "0.1"],
            "fraction_d2": ["-0.10", "-0.00", "0.00", "0.10"],
        },
        "magnitude_prefixes": [],
    },

    {
        "id": "L4",
        "distribution": "laplace",
        "params": {"loc": 0.0, "scale": 10.0},
        "prefixes": {
            "root": ["ROOT"],
            "sign": ["-"],
            "integer": ["-10", "-1", "-0", "0", "1", "10"],
            "dot": ["-10.", "-1.", "-0.", "0.", "1.", "10."],
            "fraction_d1": ["-1.0", "-0.0", "0.0", "1.0"],
            "fraction_d2": ["-1.00", "-0.00", "0.00", "1.00"],
        },
        "magnitude_prefixes": ["-1", "1"],
    },
]


EXPECTED_PARAMETER_COUNTS = {
    "normal": 6,
    "uniform": 4,
    "exponential": 4,
    "beta": 6,
    "laplace": 4,
}


EXPECTED_RUN_COUNTS = {
    "precision": 60,
    "main": 120,
    "parameter": 144,
    "prompt": 330,
    "total": 654,
}