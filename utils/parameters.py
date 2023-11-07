import json


def get_parameters_dict(gen_arch: str, disc_arch: str) -> dict:
    batch_size = 64

    if gen_arch == 'z2_rot_mnist':
        n_classes = 10
        latent_dim = 64
        use_ggan_training_loop = True
        loss_type = 'rel_avg'
        beta_1 = 0.0
        beta_2 = 0.9
        lr_g = 0.0001
        img_shape = (1, 28, 28)
    elif gen_arch == 'z2_rot_mnist_resblock':
        n_classes = 10
        latent_dim = 64
        use_ggan_training_loop = True
        loss_type = 'rel_avg'
        beta_1 = 0.0
        beta_2 = 0.9
        lr_g = 0.0001
        img_shape = (1, 28, 28)
    elif gen_arch == 'p4_rot_mnist':
        n_classes = 10
        latent_dim = 64
        use_ggan_training_loop = True
        loss_type = 'rel_avg'
        beta_1 = 0.0
        beta_2 = 0.9
        lr_g = 0.0001
        img_shape = (1, 28, 28)
    elif gen_arch == 'vanilla':
        n_classes = 10
        latent_dim = 100
        use_ggan_training_loop = False
        loss_type = 'wasserstein'
        beta_1 = 0.0
        beta_2 = 0.9
        lr_g = 0.0001
        img_shape = (1, 64, 64)
    elif gen_arch == 'vanilla_small':
        n_classes = 10
        latent_dim = 64
        use_ggan_training_loop = False
        loss_type = 'wasserstein'
        beta_1 = 0.0
        beta_2 = 0.9
        lr_g = 0.0001
        img_shape = (1, 28, 28)

    if disc_arch == 'z2_rot_mnist':
        disc_update_steps = 2
        gp_type = 'zero_centered'
        gp_strength = 0.1
        lr_d = 0.0004
    elif disc_arch == 'p4_rot_mnist':
        disc_update_steps = 2
        gp_type = 'zero_centered'
        gp_strength = 0.1
        lr_d = 0.0004
    elif disc_arch == 'vanilla':
        disc_update_steps = 5
        gp_type = 'vanilla'
        gp_strength = 10.0
        lr_d = 0.0001
    elif disc_arch == 'vanilla_small':
        disc_update_steps = 5
        gp_type = 'vanilla'
        gp_strength = 10.0
        lr_d = 0.0001

    params = {
        'gen_arch': gen_arch,
        'disc_arch': disc_arch,
        'batch_size': batch_size,
        'n_classes': n_classes,
        'latent_dim': latent_dim,
        'use_ggan_training_loop': use_ggan_training_loop,
        'loss_type': loss_type,
        'gp_type': gp_type,
        'gp_strength': gp_strength,
        'disc_update_steps': disc_update_steps,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'lr_g': lr_g,
        'lr_d': lr_d,
        'img_shape': img_shape
    }

    return params


def print_parameters(params: dict) -> None:
    for key, value in params.items():
        key = key + ': ' + '.' * (28 - len(key) - 2)
        print(f'{key : <28} {value}')
    print('\n')


def save_params_as_json(params: dict) -> None:
    json_obj = json.dumps(params, indent=4)
    with open('hyper_params.json', 'w') as outfile:
        json.dump(json_obj, outfile, indent=4)

