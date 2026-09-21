from .tabpfn import TabPFNAdapter
from .tabicl import TabICLAdapter
from .gbdt import XGBoostAdapter, CatBoostAdapter
from .tabdpt import TabDPTAdapter

MODEL_REGISTRY = {


    "xgboost": (
        XGBoostAdapter,
        {},
    ),

    "catboost": (
        CatBoostAdapter,
        {},
    ),

    "tabpfn_v2": (
        TabPFNAdapter,
        {"version": "v2"},
    ),

    "tabpfn_v2.5": (
        TabPFNAdapter,
        {"version": "v2.5"},
    ),

    "tabpfn_v2.6": (
        TabPFNAdapter,
        {"version": "v2.6"},
    ),

    "tabpfn_v3": (
        TabPFNAdapter,
        {"version": "v3"},
    ),

    "tabpfn_v3.5": (
        TabPFNAdapter,
        {"version": "v3.5"},
    ),

    "tabpfn_v3.5_fast": (
        TabPFNAdapter,
        {"version": "v3.5-fast"},
    ),

    "tabicl_v1": (
        TabICLAdapter,
        {"version": "v1"},
    ),

    "tabicl_v1.1": (
        TabICLAdapter,
        {"version": "v1.1"},
    ),

    "tabicl_v2": (
        TabICLAdapter,
        {"version": "v2"},
    ),
    #"tabdpt_v1.3": (TabDPTAdapter,{}),
    #"tabdpt": (TabDPTAdapter,{},),
}


def create_model(
    model_name: str,
    task: str,
    device: str,
    seed: int,
    **kwargs,
):
    name = model_name.lower()

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))

        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    cls, registry_kwargs = MODEL_REGISTRY[name]

    config = {
        **registry_kwargs,
        **kwargs,
    }

    return cls(
        task=task,
        device=device,
        seed=seed,
        **config,
    )