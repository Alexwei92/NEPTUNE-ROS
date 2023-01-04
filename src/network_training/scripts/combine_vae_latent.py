"""
Combine trained VAE and LatentCtrl models to reduce size
"""
import torch

from models import VanillaVAE, LatentCtrl, VAELatentCtrl

if __name__ == '__main__':

    # Load model parameter
    model_weight_config = {
        'vae_model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/vanilla_vae/vanilla_vae_model_z_1000.pt',
        'latent_ctrl_model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter1/latent_ctrl/latent_ctrl_vanilla_vae_model_z_1000.pt',
        'vae_latent_ctrl_model_path': '/media/lab/NEPTUNE2/field_outputs/imitation_learning/iter1/combined_vae_latent_ctrl_z_1000.pt',
    }

    vae_model_config = {
        'name': 'vanilla_vae',
        'in_channels': 3,
        'z_dim': 1000,
        'input_dim': 128,
    }

    latent_ctrl_model_config = {
        'name': 'latent_ctrl',
        'z_dim': vae_model_config['z_dim'],
        'extra_dim': 6,
    }

    vae_latent_ctrl_model_config = {
        'name': 'vae_latent_ctrl',
        'input_dim': vae_model_config['input_dim'],
        'in_channels': vae_model_config['in_channels'],
        'z_dim': vae_model_config['z_dim'],
        'extra_dim': latent_ctrl_model_config['extra_dim'],
    }

    try:
        # VAE
        vae_model = VanillaVAE(**vae_model_config)
        vae_model_weight = torch.load(model_weight_config['vae_model_path'])
        vae_model.load_state_dict(vae_model_weight)

        # Latent Ctrl
        latent_ctrl_model = LatentCtrl(**latent_ctrl_model_config)
        latent_ctrl_model_weight = torch.load(model_weight_config['latent_ctrl_model_path'])
        latent_ctrl_model.load_state_dict(latent_ctrl_model_weight)

        # Combine VAE Latent Ctrl
        vae_latent_ctrl_model = VAELatentCtrl(**vae_latent_ctrl_model_config)
        vae_latent_ctrl_model.Encoder = vae_model.Encoder
        vae_latent_ctrl_model.LatentFC = latent_ctrl_model.NN

        torch.save(vae_latent_ctrl_model.state_dict(), model_weight_config['vae_latent_ctrl_model_path'])
        print('File saved to %s' % model_weight_config['vae_latent_ctrl_model_path'])
        print("Successful!")

    except Exception as error:
        print('[Error]', error)
        print('Failed!')