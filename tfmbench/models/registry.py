from .tabpfn import TabPFNAdapter
from .tabicl import TabICLAdapter


MODEL_REGISTRY = {
    "tabpfn": TabPFNAdapter,
    "tabicl_v1": TabICLAdapter,
    "tabicl_v1.1": TabICLAdapter,
    "tabicl_v2": TabICLAdapter,
}

def get_supported():
    print(f"Supported models are: {MODEL_REGISTRY.keys}")

def create_model(
    model_name: str,
    task: str,
    device: str,
    seed: int,
    tabpfn_token: str,
    **kwargs,
):
    name = model_name.lower()

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    cls = MODEL_REGISTRY[name]

    if name.startswith("tabicl_"):
        version = name.removeprefix("tabicl_")

        return cls(
            task=task,
            version=version,
            device=device,
            seed=seed,
            **kwargs,
        )

    return cls(
        tabpfn_token=tabpfn_token,
        task=task,
        device=device,
        seed=seed,
        **kwargs,
    )