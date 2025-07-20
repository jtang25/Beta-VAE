import torch
from utils.brain_tumor_utils.config_parser import get_config
from utils.brain_tumor_utils.datautils import build_dataloaders
from data_processing.augmentations import get_test_transforms, get_train_transforms
from models.beta_vae import BetaVAE
from evaluation.recon_metrics import evaluate_full
from evaluation.latent_viz import generate_latent_visualizations
from evaluation.traversal import run_traversals

def load_model(weights="best"):
    cfg = get_config()
    import os
    import torch
    tag = weights
    path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_{tag}.pt"
    if not os.path.exists(path):
        path = f"{cfg.paths.models_dir}/{cfg.paths.run_id}_latest.pt"
    payload = torch.load(path, map_location="cpu")
    model = BetaVAE()
    model.load_state_dict(payload["model_state"])
    return model

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = get_train_transforms()
    test_tf = get_test_transforms()
    train_loader, test_loader = build_dataloaders(transform_train=train_tf, transform_test=test_tf)
    model = load_model("best")
    model.to(device)
    evaluate_full(model, train_loader, test_loader, device)
    generate_latent_visualizations(model, test_loader, device)
    run_traversals(model, test_loader, device)

if __name__ == "__main__":
    main()
